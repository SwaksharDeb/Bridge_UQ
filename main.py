"""
Per-sequence estimation of the BridgeUQ diffusion variance sigma, without the
amortized ScalingFactorNetwork. Every image sequence gets its own sigma^i,
estimated directly from that sequence; no training, no checkpoints.

THIS FILE is MC-EM with ONE sampler and ONE M-step, both fixed:

E-step  Langevin (ULA) on the trajectory posterior
            log p = -E(v)/gamma - S(eta)/(2 sigma)
        --em_samples chains per round, --langevin_* controls the chain.

M-step  Adam on u = log sigma (mcem.sigma_star_gd, --gd_iters / --gd_lr /
        --gd_tol / --gd_init) of the fixed-sample objective
            F(sigma) = (S_bar/2 + beta)/sigma + ((T-1)/2 + alpha + 1) log sigma,
            S(v)     = sum_{t=1}^{T-1} ((T-(t-1))/(T-t)) (v_t - mu_{t|t-1})^2.
        The samples are held fixed, so dE/dsigma = 0 and the minimizer of F is
        sigma* = (2 beta + S_bar)/(2 alpha + T + 1); the descent never evaluates
        it. Each round prints "GD M-step: ... max rel dev vs closed form=..."
        and records it in sigma_history.json -- that number is how far Adam
        stopped from sigma*.

The sigma handed to the sampler is read per --sigma_convention.

The other samplers (reverse SDE, MALA) and M-steps (closed form, total
derivative) are in main_with_different_samplers.py; mcem.py and langevin.py
still implement all of them.

Everything downstream of sigma is the learned pipeline's: the per-pixel
uncertainty is the sample std of --num_runs draws, and the metrics come from
the very same `_compute_sparsification_curves` / `_compute_risk_coverage_curves`
/ `compute_registration_metrics` in `test_fast.py`. The `s_K/batch_*.pt`
artifacts carry the same key set, so `test_fast.py` and `test.py` read them
unchanged.

Key knobs (see --help for the rest):
    --em_rounds     MC-EM rounds (sample -> gradient-descent M-step)
    --em_samples    N trajectories per round, for the sigma update
    --sample_chunk  memory knob: draw N in chunks of this size
    --num_runs      trajectories drawn AFTER sigma is final, for the UQ map
    --batch_size    image sequences per GPU pass (each keeps its own sigma^i)

Run with:
    python uncertainty_sde_combined_acdc_v3_iclr_mcem_gd/main.py \
      --backbone tlrn --batch_size 8 --em_rounds 10 --em_samples 500 \
      --sample_chunk 32 --num_runs 100

Smoke test:
    python uncertainty_sde_combined_acdc_v3_iclr_mcem_gd/main.py \
      --em_rounds 3 --em_samples 32 --sample_chunk 32 \
      --batch_size 8 --max_batches 1 --num_workers 0 \
      --optimize_only --no_training_vis --vis_every 0 --output_dir ./mcem_gd_smoke

BATCH DEPENDENCE. With the default --guidance_norm batch, grad_E is divided by
B_eff = sample_chunk * batch_size, so --batch_size CHANGES the results.
--guidance_norm per_sequence removes that dependence.
"""

import os
import sys

sys.path.append("/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2")

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import argparse
import gc
import json

import numpy as np
import torch
from tqdm import tqdm

from uncertainty_sde_combined_acdc_v3_iclr.brownian_bridge import BrownianBridgeLearnedScaling
from uncertainty_sde_combined_acdc_v3_iclr.trainer import ScalingFactorTrainer
from uncertainty_sde_combined_acdc_v3_iclr.backbones import load_backbone_and_data

# Metric / plotting helpers are imported (not copied) from test_fast.py so these
# numbers are produced by exactly the same code as the learned pipeline's.
from uncertainty_sde_combined_acdc_v3_iclr.test_fast import (
    _compute_sparsification_curves,
    _compute_risk_coverage_curves,
    compute_registration_metrics,
)

# Summary writers shared with the sigma ablations, so all four reports
# (learned / random / fixed / closed-form) are formatted identically and are
# directly comparable line for line.
from uncertainty_sde_combined_acdc_v3_iclr.main_random_sigma import (
    _write_summary,
    _write_reg_metrics,
)

from uncertainty_sde_combined_acdc_v3_iclr.losses import LTMALoss
from uncertainty_sde_combined_acdc_v3_iclr_mcem_gd.mcem import (
    mcem_refine, sigma_init, PerSequenceBridge,
    convention_scale,
    scaling_to_sigma, _expand_sf_bb,
)
from uncertainty_sde_combined_acdc_v3_iclr_mcem_gd.viz import (
    save_sigma_panel)
from uncertainty_sde_combined_acdc_v3_iclr_mcem_gd.langevin import draw_velocities

# Fixed in this file: Langevin E-step, Adam gradient-descent M-step.
SAMPLER = 'langevin'
M_STEP = 'gradient_descent'
GD_OPTIMIZER = 'adam'
# Chains per sampler call when drawing the --vis_draws training_vis samples.
VIS_DRAW_CHUNK = 256
# Colour range of the display-only interior_stretched sigma panels.
VIS_STRETCH_RANGE = (0.02, 2.0)


def interior_stretch_sigma(s_raw, lo=0.02, hi=2.0, border=2):
    """
    FOR DISPLAY ONLY. Per-sequence linear stretch of |s| = sqrt(s0^2 + s1^2)
    from its 1st/99th percentile over the image INTERIOR (a `border`-pixel
    frame excluded) onto [lo, hi], clipped; both channels rescaled by the same
    factor. Raw sigma varies by only a few percent, so this is what makes its
    spatial pattern visible (sigma_vis/.../interior_stretched). The sigma that
    is saved, sampled from and scored is always the raw one.
    """
    s = s_raw.detach().float()
    mag = s.norm(dim=1)                                                # [B, H, W]
    out = torch.empty_like(s)
    for i in range(s.shape[0]):
        q = torch.quantile(mag[i, border:-border, border:-border].flatten(),
                           torch.tensor([0.01, 0.99], device=s.device))
        p1, p99 = float(q[0]), float(q[1])
        m_new = (lo + (mag[i] - p1) / max(p99 - p1, 1e-12) * (hi - lo)).clamp(lo, hi)
        out[i] = s[i] * (m_new / mag[i].clamp_min(1e-12))
    return out.to(s_raw.dtype)


def main():
    parser = argparse.ArgumentParser(
        description='BridgeUQ with MC-EM diffusion variance: Langevin E-step, '
                    'gradient-descent (Adam) M-step (no learned scaling network)'
    )
    # --- what to run on (mirrors main.py / main_fixed_sigma.py) ---
    parser.add_argument('--backbone',
                        choices=['tlrn', 'ltma', 'tm', 'voxelmorph', 'vmplus'],
                        default='tlrn',
                        help='Pretrained registration backbone (default: tlrn)')
    parser.add_argument('--combine_acdc', action='store_true', default=True,
                        help='Combine original 1200 + ACDC 862 training samples (default: True)')
    parser.add_argument('--no_combine_acdc', dest='combine_acdc', action='store_false',
                        help='Use only the original 1200 training samples')
    parser.add_argument('--use_train', action='store_true', default=False,
                        help='Evaluate on the TRAIN split instead of the TEST split')
    parser.add_argument('--batch_size', type=int, default=8, #128
                        help='Loader batch size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Override loader worker count. Use 0 on nodes with strict '
                             'overcommit (vm.overcommit_memory=2) where forking fails.')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Stop after this many batches (quick smoke runs).')
    parser.add_argument('--loss', default='mse',
                        choices=['mse', 'l1', 'ncc', 'gncc', 'ssim', 'msssim',
                                 'mi', 'nmi', 'ngf', 'mind', 'deepsim', 'deepsim_mse'],
                        help='Only used to resolve the workdir suffix, matching main.py. '
                             'No loss is evaluated here -- nothing is trained.')
    parser.add_argument('--workdir', type=str, default=None,
                        help='Override config.model.params.workdir (applied before the '
                             '--loss suffix).')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where results are written. Default: {workdir}/visualization/'
                             'uncertainty_sde_combined_acdc_v3_iclr_mcem_gd')

    parser.add_argument('--langevin_steps', type=int, default=100,
                        help='ULA iterations per draw (default: 100). Each costs '
                             'one energy gradient, so this multiplies sampling '
                             'cost relative to the 14-step reverse SDE.')
    parser.add_argument('--langevin_precondition', choices=['none', 'diag'],
                        default='none',
                        help="Per-voxel step of the Langevin sampler. 'none' "
                             "(default): eps = step_size * sigma. 'diag': "
                             "eps = step_size * M, M = 1/(1/sigma + "
                             "2 g |grad I|^2 / (gamma H W)) -- smaller steps "
                             "where the image makes the likelihood stiff "
                             "(edges), full steps in flat regions "
                             "(langevin._diag_precond). Constant during a draw, "
                             "so the target is unchanged. This is what keeps a "
                             "WHITE chain (--langevin_noise_smooth_width 0) "
                             "stable at small gamma: measured at gamma 5e-6, "
                             "step_size 0.05, --reg_weight 0, it converges in "
                             "~100 steps (kappa ~1 in flat regions) where the "
                             "K=2 sampler reaches only kappa 0.18 after 3200.")
    parser.add_argument('--langevin_burn_in', type=int, default=99,
                        help="States discarded from the "
                             "front of each chain. Samples per chain = "
                             "(--langevin_steps - burn_in) / --langevin_thin, "
                             "all of which feed S_bar, so the E-step averages "
                             "N_eff = --em_samples x samples trajectories. "
                             "Default: --langevin_steps - 1, i.e. one sample "
                             "per chain -- the final state, exactly the "
                             "previous behaviour. The chain's slowest mode "
                             "relaxes in ~400 steps at step_size 0.05 "
                             "(langevin.py, langevin_chain), and at K width 2 "
                             "100 steps reach only ~10%% of the stationary "
                             "spread, so a real burn-in should be several "
                             "hundred steps. Mirrors uncertainty_sde_acdc_"
                             "combined_iclr_2/main_langevin.py's flag. "
                             "Ignored by the post-loop UQ draws.")
    parser.add_argument('--langevin_thin', type=int, default=1,
                        help="Keep every k-th "
                             "post-burn-in state (default 1, keep all). "
                             "Consecutive states are strongly correlated, so a "
                             "larger thin gives closer-to-independent samples "
                             "for the same count, at the cost of a longer "
                             "chain.")
    parser.add_argument('--langevin_step_size', type=float, default=0.02, #0.02
                        help='eps = this * sigma. Stability needs it < 2*rho with '
                             'rho >= 0.5, so keep it <= 0.5 (default: 0.1).')
    parser.add_argument('--langevin_init', choices=['prior', 'zero'],
                        default='zero',
                        help="ULA initialization: 'zero' (default) starts at the "
                             "registered velocity v = v_hat, 'prior' draws from "
                             "the exact bridge prior.")
    parser.add_argument('--langevin_noise_smooth_width', type=float, default=2,  #0.2
                        help='Width w of the Gaussian kernel K that turns on the '
                             'SMOOTHED Langevin sampler. Any w > 0 enables it; 0 is '
                             'the plain isotropic update. Default 4.0, i.e. smoothing '
                             'is ON unless you pass 0. The preconditioner is then the '
                             'matrix A = sigma^(1/2) K sigma^(1/2) = S K S and the '
                             'step becomes '
                             'eta <- eta + eps*A(grad log p) + sqrt(2*eps)*A^(1/2)(z). '
                             'The drift is sqrt(eps)*K(sqrt(eps)*grad log p), i.e. S '
                             'on BOTH sides of the blur at width w, and the noise is '
                             'sqrt(2*eps)*K^(1/2)(z) -- K^(1/2) is the Gaussian of '
                             'width w/sqrt(2) because G_a*G_a = G_{a*sqrt2}. Since '
                             'A = (S*K^(1/2))(K^(1/2)*S), that noise has Cov = 2*eps*A, '
                             'matching the drift operator, which is what keeps the '
                             'chain targeting the posterior: one earlier version '
                             'blurred only the noise and left the drift isotropic, and '
                             'a later one used S^2*K for the drift, which is not '
                             'symmetric once sigma varies per voxel -- both chains '
                             'targeted nothing well-defined. Because A preconditions '
                             'rather '
                             'than reweights, the target density is UNCHANGED -- '
                             'sigma stays comparable across widths, and the M-step '
                             'consistency still holds. Distinct from '
                             '--langevin_smooth_width, which smooths the FIELD '
                             '(v = v_hat + K*xi) and does change the target; the two '
                             'are independent and may be combined.')
    parser.add_argument('--gd_iters', type=int, default=500,
                        help='Max Adam iterations per M-step (default: 500). '
                             'Cheap: well under a second on a [8,2,128,128] '
                             'sigma.')
    parser.add_argument('--gd_lr', type=float, default=0.05,
                        help='Adam lr on log sigma for the M-step '
                             '(default: 0.05).')
    parser.add_argument('--gd_tol', type=float, default=1e-9,
                        help='Stop the M-step early '
                             'once max |dF/du| < tol over every voxel (float64 '
                             'solve; default 1e-9). 0 runs all --gd_iters.')
    parser.add_argument('--gd_init', choices=['previous', 'prior'], default='previous',
                        help="Where each M-step's "
                             "descent starts. 'previous' (default) warm-starts "
                             "from the previous round's sigma; 'prior' restarts "
                             "from the constant sigma^1 = 2 beta/(2 alpha+T+1) "
                             "every round.")

    # --- the M-step objective ---
    parser.add_argument('--alpha', type=float, default=1e-4,
                        help='Inverse-Gamma shape alpha (default: 1e-4, the paper '
                             "Table 2 optimum). Enters sigma* only through the "
                             'denominator 2*alpha + T + 1.')
    parser.add_argument('--beta', type=float, default=300.0,
                        help='Inverse-Gamma scale beta (default: 3.0, the paper Table 2 '
                             'optimum). Sets sigma^1 = 2*beta/(2*alpha+T+1) and the '
                             'prior-only fixed point beta/(alpha+1).')
    parser.add_argument('--sigma_convention', choices=['code_s', 'paper_sigma'],
                        default='code_s',
                        help="Which quantity the closed form solves for. "
                             "'code_s' (default) solves directly for the "
                             "`scaling_factors` s, matching the delta = s*rho_t "
                             "used by losses.py's bridge_norm and the variable its "
                             "inv_gamma_prior is applied to -- i.e. the convention "
                             "the learned network was trained under. Gives "
                             "s^1 = 2*beta/(2*alpha+T+1) = 0.400 and settles near "
                             "0.47, inside the ~0.4-1.5 range the learned network "
                             "produces. 'paper_sigma' solves for the paper's "
                             "diffusion variance and converts by s = sigma*T/2 "
                             "(s^1 = 2.80); NOTE it does not converge at these "
                             "hyperparameters -- the Euler-Maruyama sampler "
                             "over-disperses by a measured kappa=1.2454, putting "
                             "the EM slope at 1.079 > 1. See mcem.convention_scale.")
    parser.add_argument('--em_rounds', type=int, default=10,
                        help='MC-EM rounds R: sample -> gradient-descent M-step '
                             '(default: 20). Under the prior alone the map contracts '
                             'at (T-1)/(2*alpha+T+1) ~ 0.87 per round at T=14.')
    parser.add_argument('--em_samples', type=int, default=500, #1
                        help='N posterior trajectories per EM round, averaged into '
                             'S_bar for the M-step (default: 32).')
    parser.add_argument('--border', type=int, default=2,
                        help='Exclude this many pixels at the image border from '
                             'the sigma estimate (default 2; 0 = off). After every '
                             'M-step the border pixels take the value of the '
                             'nearest interior pixel, so the sampler never sees '
                             'the bright ring the M-step leaves there. '
                             'The border is still scored by '
                             'the metrics here; test_mcem.py --border crops it.')
    parser.add_argument('--sample_chunk', type=int, default=32, #12
                        help='Draw the N EM trajectories this many at a time, so peak '
                             'memory grows as sample_chunk*batch_size instead of '
                             'N*batch_size (default: 12). 0 = all N at once.')

    # --- SDE / UQ settings (must match main.py to be comparable) ---
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Posterior reverse-SDE trajectories per subject drawn AFTER '
                             'the EM loop, for the uncertainty map (default: 100)')
    parser.add_argument('--num_diffusion_steps', type=int, default=14,
                        help='Diffusion steps T (default: 14, matching main.py)')
    parser.add_argument('--gamma', type=float, default=5e-5,  #5e-6, 5e-5
                        help="Likelihood temperature in the posterior SDE. The "
                             "correction is (guidance_scale/gamma)*sigma_sq*grad_E, "
                             "and under --guidance_norm batch the inherited grad_E "
                             "carries a 1/B_eff factor, so the EFFECTIVE strength "
                             "there is (1/gamma)/B_eff. Removing that factor means "
                             "gamma must grow by the same B_eff or the SDE diverges "
                             "(measured: gamma=5e-5 per-sequence blows sigma up to "
                             "~1e17 and NaNs at reverse step t=3). Default: 5e-5 for "
                             "--guidance_norm batch; 5e-5*--gamma_ref_batch for "
                             "per_sequence, which reproduces the effective guidance "
                             "v3_iclr's UQ sampling had at its default batch_size. "
                             "This is a STARTING POINT -- gamma is now a real "
                             "batch-independent temperature and is worth retuning.")
    parser.add_argument('--gamma_ref_batch', type=int, default=128,
                        help="Only used to pick the default --gamma under "
                             "--guidance_norm per_sequence: gamma = 5e-5 * this. "
                             "128 is v3_iclr main.py's default batch_size, i.e. the "
                             "B_eff its post-refinement UQ sampling ran at "
                             "(main.py:837-845, untiled). Ignored if --gamma is "
                             "given explicitly.")
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                        help='Energy-gradient guidance scale (default: 1.0)')
    parser.add_argument('--reg_weight', type=float, default=0.0, #0.1
                        help='Regularization weight inside the energy gradient (default: 0.1)')
    parser.add_argument('--guidance_norm', choices=['per_sequence', 'batch'],
                        default='batch',
                        help="How the Eq. 13 likelihood score is normalized. "
                             "'per_sequence' (default) gives every image sequence "
                             "the score it would have if run alone, so the result "
                             "does not depend on batch composition. 'batch' keeps "
                             "the inherited behaviour, where F.mse_loss divides by "
                             "the whole stack and the score is therefore weaker by "
                             "a factor B_eff = sample_chunk*batch_size (verified "
                             "exactly 1/B_eff by measurement). Use 'batch' only to "
                             "reproduce the existing v3_iclr pipeline. NOTE: "
                             "'per_sequence' makes the guidance B_eff times "
                             "stronger at the same --gamma, so gamma needs scaling "
                             "up accordingly (see --gamma).")
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed for the reverse-SDE noise (default: 0)')

    # --- metrics ---
    parser.add_argument('--error_metric', choices=['similarity', 'segmentation'],
                        default='segmentation',
                        help='Per-pixel error used for nAUSE / nAURC. "similarity" uses '
                             '(warped - target)^2; "segmentation" uses per-pixel binary '
                             'disagreement of the warped LV seg (default: similarity).')
    parser.add_argument('--sigma_vis_style', choices=['overlay', 'plain'],
                        default='plain',
                        help="How sigma_vis panels draw |s|. 'overlay' (default) "
                             "matches v3_iclr's training figure: source frame in "
                             "gray with |s| in viridis at alpha 0.7 on top. "
                             "'plain' draws |s| alone. The overlay lets the "
                             "image's anatomy show through even where sigma is "
                             "flat, so use 'plain' to judge whether the "
                             "structure is really in sigma.")
    parser.add_argument('--vis_every', type=int, default=1,
                        help='Save the sigma panel every k EM rounds / td steps '
                             'to sigma_vis/batch_XXX/round_XXXX.png (default 1 = '
                             'every round). 0 = off; the final '
                             'sigma_vis/batch_XXX.png is written either way.')
    parser.add_argument('--no_training_vis', dest='training_vis', action='store_false',
                        default=True,
                        help='Skip the per-round 7-row figure in the v3_iclr '
                             'training_vis layout (trainer.visualize_training_step), '
                             'written every --vis_every rounds to '
                             '<output_dir>/training_vis/batch_NNN/training_vis_iterNNNN.png.')
    parser.add_argument('--vis_subject_idx', type=int, default=None,
                        help='Subject (index within the batch) shown in the '
                             'training_vis figure. Default: one drawn at random '
                             'per batch and kept for all of that batch\'s rounds.')
    parser.add_argument('--vis_draws', type=int, default=64,
                        help='Posterior draws for that subject per training_vis '
                             'figure. Row |v_t| shows the first draw; row '
                             '|std_tilde| shows the per-frame std across all '
                             'draws (Langevin has no analytic std_tilde). At 8 '
                             'the per-frame std is mostly sampling noise '
                             '(measured corr with sigma 0.74 per frame); 64 '
                             'gives 0.95 (default: 64).')
    parser.add_argument('--optimize_only', action='store_true', default=False,
                        help='Run the EM loop, save s^R, and stop. Skips the post-loop UQ '
                             'sampling, the per-subject metrics and all plots -- for '
                             'timing runs. test_fast.py can evaluate s^R afterwards.')
    parser.add_argument('--no_reg_metrics', dest='compute_reg_metrics',
                        action='store_false', default=True,
                        help='Skip the Dice / HD95 / RMSE / neg-Jac computation.')
    args = parser.parse_args()

    # Resolve gamma against the chosen normalization (see --gamma).
    _GAMMA_BATCH = 0.0001 / 2
    if args.gamma is None:
        if args.guidance_norm == 'per_sequence':
            args.gamma = _GAMMA_BATCH * args.gamma_ref_batch
            why = f"5e-5 * {args.gamma_ref_batch} (per-sequence equivalent)"
        else:
            args.gamma = _GAMMA_BATCH
            why = "v3_iclr default"
        print(f"gamma not given -> {args.gamma:g}  ({why})")

    _nsw = args.langevin_noise_smooth_width
    if _nsw and _nsw > 0:
        print(f"K-preconditioned Langevin: A = S K S, S = sqrt(step*sigma), "
              f"K width {_nsw:g}")

    NUM_RUNS = args.num_runs
    device = 'cuda'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------
    # Load model and data
    # ------------------------------------------------------------------
    model, train_loader, test_loader, config = load_backbone_and_data(
        backbone=args.backbone, combine_acdc=args.combine_acdc,
    )
    print(f"Using backbone: {args.backbone.upper()}")
    if args.workdir is not None:
        config.model.params.workdir = args.workdir
        print(f"Workdir overridden: {config.model.params.workdir}")
    if args.loss != 'mse':
        config.model.params.workdir = f"{config.model.params.workdir}_{args.loss}"
    print(f"Workdir: {config.model.params.workdir}")

    from torch.utils.data import DataLoader
    base_loader = train_loader if args.use_train else test_loader
    data_split = "TRAIN" if args.use_train else "TEST"
    workers = (args.num_workers if args.num_workers is not None
               else getattr(base_loader, 'num_workers', 4))
    # shuffle=False: global_subj_idx below is the s^R key whenever the loader
    # yields no per-sample 'path', so the ordering has to be deterministic and
    # independent of batch_size -- same contract as main.py.
    data_loader = DataLoader(
        base_loader.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=getattr(base_loader, 'pin_memory', True),
        collate_fn=getattr(base_loader, 'collate_fn', None),
    )
    print(f"Using {data_split} set | batch_size={args.batch_size} | workers={workers}")

    first_batch = next(iter(data_loader))
    first_series = first_batch['series'].cuda()
    first_lv_segs = first_batch.get('lv_segs')
    first_lv_segs = (first_lv_segs.cuda() if first_lv_segs is not None
                     else torch.zeros_like(first_series[:, 0:1]).cuda())
    with torch.no_grad():
        _, first_v_series, _, _, _ = model.sequence_register_no_avg_lowf_addlatentf(
            first_series, first_lv_segs
        )
    num_time_steps = len(first_v_series)
    img_size = first_series.shape[-1]
    print(f"Image size: {img_size}, time steps: {num_time_steps}")
    del first_series, first_lv_segs, first_v_series
    torch.cuda.empty_cache()

    # T in Eq. 10 is the frame count, and T in `sigma_sq = 2*s/T` is the
    # diffusion-step count. They coincide here (15 cardiac frames -> T = 14 =
    # --num_diffusion_steps), which is what makes
    # diffusion_timestep_to_cardiac_frame the identity. Refuse to run if they
    # ever diverge rather than silently mixing two different T's.
    T_frames = num_time_steps          # = len([zeros] + v_series) - 1
    if T_frames != args.num_diffusion_steps:
        raise ValueError(
            f"Frame count T={T_frames} (from the data) != --num_diffusion_steps="
            f"{args.num_diffusion_steps}. The closed form uses T from Eq. 10 while "
            f"the bridge uses sigma_sq = 2*s/T_diff; the sigma <-> s conversion is "
            f"only well defined when the two agree. Pass "
            f"--num_diffusion_steps {T_frames}.")

    # ------------------------------------------------------------------
    # SDE components. NOTE: no ScalingFactorNetwork and no optimizer -- the
    # trainer is used purely as a thin wrapper around the reverse SDE.
    # ------------------------------------------------------------------
    bridge_cls = (PerSequenceBridge if args.guidance_norm == 'per_sequence'
                  else BrownianBridgeLearnedScaling)
    brownian_bridge = bridge_cls(
        num_diffusion_steps=args.num_diffusion_steps,
        img_size=img_size,
        reg_weight=args.reg_weight,
    )
    # Grad-free Langevin: the E-step and the post-loop UQ draws.
    langevin_kw = dict(n_steps=args.langevin_steps,
                       step_size=args.langevin_step_size,
                       init=args.langevin_init,
                       precondition=args.langevin_precondition,
                       noise_smooth_width=args.langevin_noise_smooth_width)
    # Burn-in / thinning for the E-step; every chain restarts from
    # --langevin_init each round.
    burn = (args.langevin_steps - 1 if args.langevin_burn_in is None
            else int(args.langevin_burn_in))
    if not 0 <= burn < args.langevin_steps:
        raise SystemExit(
            f"--langevin_burn_in ({burn}) must be in [0, --langevin_steps "
            f"({args.langevin_steps})), or the chain has no post-burn-in "
            f"states to sample.")
    n_per_chain = max(1, (args.langevin_steps - burn) // max(1, args.langevin_thin))
    chain_kw = dict(burn_in=burn, thin=args.langevin_thin, n_samples=n_per_chain)
    print(f"Langevin chain: {args.langevin_steps} steps, burn-in {burn}, "
          f"thin {args.langevin_thin} -> {n_per_chain} sample(s)/chain, "
          f"N_eff {args.em_samples * n_per_chain}/round")
    gd_kw = dict(iters=args.gd_iters, lr=args.gd_lr, tol=args.gd_tol,
                 init=args.gd_init, optimizer=GD_OPTIMIZER)
    loss_fn = LTMALoss(lambda_sim=1.0, lambda_reg=0.0,
                       lambda_scale=args.beta, alpha_scale=args.alpha,
                       lambda_low_structure=0.0, sim_type=args.loss,
                       gamma=args.gamma)
    trainer = ScalingFactorTrainer(
        scaling_network=None,
        brownian_bridge=brownian_bridge,
        loss_fn=loss_fn,
        device=device,
        img_size=img_size,
        gamma=args.gamma,
        guidance_scale=args.guidance_scale,
    )

    if args.output_dir:
        out_root = args.output_dir
    else:
        out_root = (f"{config.model.params.workdir}/visualization/"
                    f"uncertainty_sde_combined_acdc_v3_iclr_mcem_gd")
    os.makedirs(out_root, exist_ok=True)
    s_K_dir = os.path.join(out_root, "s_K")
    os.makedirs(s_K_dir, exist_ok=True)

    with open(os.path.join(out_root, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    sigma1 = sigma_init(args.alpha, args.beta, T_frames)
    contraction = (T_frames - 1) / (2 * args.alpha + T_frames + 1)
    print(f"\n{'=' * 70}")
    print(f"MC-EM, LANGEVIN + GRADIENT-DESCENT M-STEP ({data_split})".center(70))
    print(f"{'=' * 70}")
    print(f"  M-step          : Adam on log sigma, <= {args.gd_iters} iters, "
          f"lr {args.gd_lr:g}, tol {args.gd_tol:g}, init {args.gd_init}")
    print(f"  its minimizer   : sigma* = (2*beta + S_bar) / (2*alpha + T + 1), "
          f"evaluated only for the reported deviation")
    print(f"  alpha={args.alpha:g}, beta={args.beta:g}, T={T_frames}")
    conv_k = convention_scale(args.sigma_convention, args.num_diffusion_steps)
    print(f"  convention      : {args.sigma_convention} "
          f"(s = solved_variable * {conv_k:g})")
    print(f"  sigma^1 = 2*beta/(2*alpha+T+1) = {sigma1:.6f}  (spatially constant)")
    print(f"  -> scaling s^1 = {sigma1 * conv_k:.6f}"
          f"   [learned network's |s| spans ~0.4-1.5]")
    print(f"  EM              : R={args.em_rounds}, N={args.em_samples} "
          f"(chunk {args.sample_chunk}) — PER SEQUENCE "
          f"(each sequence has its own sigma^i)")
    print(f"  guidance_norm   : {args.guidance_norm}"
          + ("  (score is what the sequence would get alone; batch "
             "composition does not affect it)"
             if args.guidance_norm == 'per_sequence'
             else f"  (inherited 1/B_eff shrinkage, B_eff="
                  f"{max(1, args.sample_chunk) * args.batch_size})"))
    print(f"  prior-only fixed point beta/(alpha+1) = "
          f"{args.beta / (args.alpha + 1):.6f}, contraction {contraction:.4f}/round")
    print(f"  sampler         : langevin, T={args.num_diffusion_steps}, gamma={args.gamma}, "
          f"guidance={args.guidance_scale}, reg_weight={args.reg_weight}")
    print(f"  UQ trajectories : {NUM_RUNS} per subject")
    print(f"  error_metric    : {args.error_metric}")
    print(f"  output          : {out_root}")
    print(f"{'=' * 70}\n")

    # ------------------------------------------------------------------
    # Batch loop. Per batch: ONE registration forward pass -> MC-EM loop for
    # sigma -> NUM_RUNS posterior draws -> per-subject curves.
    # ------------------------------------------------------------------
    all_ids, all_spars, all_rc, all_reg = [], [], [], []
    all_histories = []
    global_subj_idx = 0

    total_batches = len(data_loader)
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)

    for batch_idx, batch in enumerate(data_loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        print(f"\nBatch {batch_idx + 1}/{total_batches}")

        series = batch['series'].cuda()
        lv_segs = batch.get('lv_segs')
        lv_segs = (lv_segs.cuda() if lv_segs is not None
                   else torch.zeros_like(series[:, 0:1]).cuda())
        file_paths = batch.get('path', [None] * series.shape[0])

        with torch.no_grad():
            Sdef_series, v_series, _, _, _ = \
                model.sequence_register_no_avg_lowf_addlatentf(series, lv_segs)

        batch_size = series.shape[0]
        source_img = series[:, 0:1]
        target_imgs = [series[:, t:t + 1] for t in range(1, series.shape[1])]
        v_series_bb = [torch.zeros_like(v_series[0])] + v_series
        num_cardiac_frames = len(v_series_bb)

        # Per-round sigma panel, same rendering as the final one below.
        round_vis_dir = os.path.join(out_root, "sigma_vis", f"batch_{batch_idx:03d}")
        round_ids = [os.path.basename(str(fp)) if fp is not None
                     else f"subj_{global_subj_idx + i:05d}"
                     for i, fp in enumerate(file_paths if isinstance(file_paths, list)
                                            else [file_paths] * batch_size)]
        round_label = 'round'

        training_vis_dir = os.path.join(out_root, "training_vis", f"batch_{batch_idx:03d}")
        vis_i = (int(np.random.randint(0, batch_size)) if args.vis_subject_idx is None
                 else min(max(0, args.vis_subject_idx), batch_size - 1))
        # The sampler's sigma for scaling s: the inverse of the conversion the
        # M-step applied before handing s to the callback.
        vis_conv = args.sigma_convention

        def _save_training_vis(r, s_raw):
            """v3_iclr's 7-row figure for subject vis_i at this round's sigma.

            Rows: target, warped (one draw), |v_hat|, |s| overlay, std across
            --vis_draws draws, |v_t| (one draw), |warp - target|. Uses
            trainer.visualize_training_step unchanged; the draws are made for
            subject vis_i only and scattered into batch-sized tensors so the
            title's subject index is the real one.
            """
            s_raw = s_raw.detach()
            n = max(1, int(args.vis_draws))
            sf_bb_all = _expand_sf_bb(s_raw, num_cardiac_frames)[0]
            sl = slice(vis_i, vis_i + 1)
            sigma_i = scaling_to_sigma(s_raw[sl], args.num_diffusion_steps, vis_conv)
            # Drawn in chunks of at most VIS_DRAW_CHUNK chains so a large
            # --vis_draws does not run out of memory; the per-frame mean and
            # variance are accumulated in float64 over all chunks. The Langevin
            # energy is per chain, so chunking does not change the draws' law.
            first, s1, s2 = None, None, None
            done = 0
            while done < n:
                n_c = min(VIS_DRAW_CHUNK, n - done)
                rep = lambda t: t[sl].repeat(n_c, *([1] * (t.dim() - 1)))
                v_draws, _ = draw_velocities(
                    SAMPLER, trainer, [rep(v) for v in v_series_bb],
                    rep(sf_bb_all), rep(source_img), [rep(t) for t in target_imgs],
                    num_samples=n_c, sigma=sigma_i.repeat(n_c, 1, 1, 1),
                    langevin_kw=langevin_kw)
                v_draws = [v.detach() for v in v_draws]      # each [n_c, C, H, W]
                if first is None:
                    first = [v[0].clone() for v in v_draws]
                    s1 = [torch.zeros_like(v[0], dtype=torch.float64) for v in v_draws]
                    s2 = [torch.zeros_like(v[0], dtype=torch.float64) for v in v_draws]
                for k, v in enumerate(v_draws):
                    s1[k] += v.double().sum(dim=0)
                    s2[k] += (v.double() ** 2).sum(dim=0)
                done += n_c
                del v_draws
            with torch.no_grad():
                v_t_list, var_list, warped = [], [], []
                for k in range(len(first)):
                    full = torch.zeros_like(v_series_bb[0]); full[vis_i] = first[k]
                    var = torch.zeros_like(full)
                    if n > 1:                           # unbiased; row uses sqrt(var0 + var1)
                        var[vis_i] = ((s2[k] - s1[k] ** 2 / n) / (n - 1)).clamp_min(0).to(full.dtype)
                    v_t_list.append(full)
                    var_list.append(var)
                    warped.append(trainer.warp_image(source_img, full))
                trainer.visualize_training_step(
                    sf_bb_all, var_list, v_series_bb, v_t_list,
                    source_img, target_imgs, warped,
                    training_vis_dir, epoch=r - 1, subject_idx=vis_i)

        def _save_round_vis(r, s_raw):
            # Raw per-round values, EVERY round (independent of --vis_every):
            # s as the sampler consumed it, and sigma in the M-step's units.
            os.makedirs(os.path.join(round_vis_dir, "values"), exist_ok=True)
            torch.save({
                'round': r,
                'subject_ids': round_ids,
                'scaling_factors_raw': s_raw.detach().cpu(),
                'sigma': scaling_to_sigma(s_raw.detach(), args.num_diffusion_steps,
                                          vis_conv).cpu(),
                'convention': vis_conv,
            }, os.path.join(round_vis_dir, "values", f"round_{r:04d}.pt"))
            if args.vis_every <= 0 or r % args.vis_every:
                return
            save_sigma_panel(
                s_raw, source_img, os.path.join(round_vis_dir, f"round_{r:04d}.png"),
                subject_ids=round_ids, per_panel=True,
                overlay=args.sigma_vis_style == 'overlay',
                title=f"sigma |s|  (batch {batch_idx}, {M_STEP}, "
                      f"{round_label} {r})")
            # Same round, stretched on the INTERIOR 1-99 percentile (border ring
            # excluded) for display only, on a fixed colour scale.
            lo, hi = VIS_STRETCH_RANGE
            save_sigma_panel(
                interior_stretch_sigma(s_raw, lo=lo, hi=hi), source_img,
                os.path.join(round_vis_dir, "interior_stretched", f"round_{r:04d}.png"),
                subject_ids=round_ids, vmin=lo, vmax=hi,
                overlay=args.sigma_vis_style == 'overlay',
                title=f"sigma |s|, interior 1-99% stretched to [{lo:g}, {hi:g}]  "
                      f"(batch {batch_idx}, {M_STEP}, {round_label} {r})")
            if args.training_vis:
                _save_training_vis(r, s_raw)

        # ---- MC-EM in place of the network's s^K: Langevin E-step, Adam
        # gradient-descent M-step ----
        # No torch.no_grad(): the sampler calls torch.autograd.grad internally
        # for the energy gradient.
        try:
            (scaling_factors_bb, sigma, v_bar_list, u_bar_list,
             history, rounds_run) = mcem_refine(
                trainer, source_img, target_imgs, v_series_bb,
                rounds=args.em_rounds,
                num_samples=args.em_samples,
                sample_chunk=args.sample_chunk,
                alpha=args.alpha,
                beta=args.beta,
                tol=0.0,
                convention=args.sigma_convention,
                sampler=SAMPLER,
                langevin_kw=langevin_kw,
                round_callback=_save_round_vis,
                chain_kw=chain_kw,
                m_solver=M_STEP,
                gd_kw=gd_kw,
                border=args.border,
                device=device,
            )
        except RuntimeError as e:
            if 'NaN' not in str(e):
                raise
            raise RuntimeError(
                f"{e}\n\n"
                f"The posterior SDE diverged. The usual cause is guidance that is "
                f"too strong for --gamma: the correction is "
                f"(guidance_scale/gamma)*sigma_sq*grad_E, and sigma grows each EM "
                f"round, so an over-strong term is self-amplifying (sigma -> 1e17 "
                f"-> NaN). Current: guidance_norm={args.guidance_norm}, "
                f"gamma={args.gamma:g}, guidance_scale={args.guidance_scale:g}. "
                f"Try raising --gamma (10x steps), lowering --guidance_scale, or "
                f"lowering --beta (which sets the prior fixed point sigma -> "
                f"beta/(alpha+1) and hence the scale sigma_sq operates at)."
            ) from e
        all_histories.append({'batch_idx': batch_idx,
                              'history': history,
                              'rounds_per_sequence': rounds_run.tolist()})
        mean_u = float(torch.stack([u.mean() for u in u_bar_list]).mean().item())

        # ---- Persist s^R, same key set as main.py:797-810 so test_fast.py /
        # test.py consume it unchanged, plus the MC-EM specifics. ----
        subject_paths_for_save = []
        for i in range(batch_size):
            fp = file_paths[i] if isinstance(file_paths, list) else file_paths
            subject_paths_for_save.append(
                str(fp) if fp is not None else f"subj_{global_subj_idx + i:05d}"
            )

        s_K_path = os.path.join(s_K_dir, f"batch_{batch_idx:03d}.pt")
        torch.save({
            'batch_idx': batch_idx,
            'inner_iterations': int(rounds_run.max()),  # EM rounds actually run
            'subject_paths': subject_paths_for_save,
            'scaling_factors_raw': scaling_factors_bb[:, 1].detach().cpu(),
            'gamma': args.gamma,
            'guidance_scale': args.guidance_scale,
            'num_diffusion_steps': args.num_diffusion_steps,
            'num_cardiac_frames': num_cardiac_frames,
            'loss_dict': None,                       # nothing is optimized
            'checkpoint': None,                      # no network, no checkpoint
            'mean_u': mean_u,
            # --- MC-EM specifics ---
            'm_step': M_STEP,
            'sampler': SAMPLER,
            'gd_kw': gd_kw,
            'sigma': sigma.detach().cpu(),           # PAPER units (s = sigma*T/2)
            'alpha': args.alpha,
            'beta': args.beta,
            'sigma_convention': args.sigma_convention,
            'em_rounds': int(rounds_run.max()),
            'em_rounds_per_sequence': rounds_run,   # [B] -- per image sequence
            'sigma_history': history,
            'guidance_norm': args.guidance_norm,
            # No post-processing: scaling_factors_raw above IS the M-step's sigma.
            # Keys kept (as None) so readers of older s_K files still work.
            'scaling_factors_em_raw': None,
            'sigma_post': None,
        }, s_K_path)
        print(f"  s^R saved: {s_K_path}  (mean_u={mean_u:.4f}, "
              f"sigma mean={float(sigma.mean()):.6f})")

        # Sigma panel in the same form as row 4 of v3_iclr's training figure,
        # so the closed-form s can be compared directly against the learned one.
        s_raw_final = scaling_factors_bb[:, 1]          # [B, 2, H, W]
        vis_stats = save_sigma_panel(
            s_raw_final, source_img,
            os.path.join(out_root, "sigma_vis", f"batch_{batch_idx:03d}.png"),
            subject_ids=[os.path.basename(p) for p in subject_paths_for_save],
            per_panel=True, overlay=args.sigma_vis_style == 'overlay',
            title=(f"Gradient-descent M-step sigma  |s|  (batch {batch_idx}, "
                   f"{args.sigma_convention}, alpha={args.alpha:g}, "
                   f"beta={args.beta:g}, gamma={args.gamma:g})"),
        )
        # Final EM sigma, interior 1-99% stretched (display only).
        _lo, _hi = VIS_STRETCH_RANGE
        save_sigma_panel(
            interior_stretch_sigma(s_raw_final, lo=_lo, hi=_hi),
            source_img,
            os.path.join(out_root, "sigma_vis", f"batch_{batch_idx:03d}_interior_stretched.png"),
            subject_ids=[os.path.basename(p) for p in subject_paths_for_save],
            vmin=_lo, vmax=_hi, overlay=args.sigma_vis_style == 'overlay',
            title=f"sigma |s|, interior 1-99% stretched to [{_lo:g}, {_hi:g}]  (batch {batch_idx})")
        print(f"  |s| = sqrt(s0^2+s1^2): "
              f"[{vis_stats['mag_min']:.3f}, {vis_stats['mag_max']:.3f}] "
              f"mean {vis_stats['mag_mean']:.3f}, "
              f"within-subject std {vis_stats['mag_std_within']:.3f}   "
              f"(learned reference: ~0.4-1.5)")

        if args.optimize_only:
            global_subj_idx += batch_size
            del (series, lv_segs, Sdef_series, v_series, v_series_bb,
                 source_img, target_imgs, scaling_factors_bb, sigma,
                 v_bar_list, u_bar_list)
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # ---- Posterior Langevin sampling for the UQ map ----
        # The UQ draws must disperse exactly as the E-step's did: sigma is
        # already in --sigma_convention units, which mcem_refine passed to the
        # sampler unchanged.
        sigma_uq = sigma
        sampled_velocities = {i: [] for i in range(num_cardiac_frames)}
        for run in tqdm(range(NUM_RUNS), desc="  UQ runs", leave=False):
            v_t_list, _ = draw_velocities(
                SAMPLER, trainer, v_series_bb, scaling_factors_bb,
                source_img, target_imgs, num_samples=1,
                sigma=sigma_uq, langevin_kw=langevin_kw,
            )
            for frame_idx in range(num_cardiac_frames):
                sampled_velocities[frame_idx].append(v_t_list[frame_idx].detach().clone())
            if (run + 1) % 10 == 0:
                torch.cuda.empty_cache()

        # ---- Per-subject nAUSE / nAURC ----
        batch_ids, batch_spars, batch_rc, batch_reg = [], [], [], []
        for subj_idx in range(batch_size):
            fname = file_paths[subj_idx] if isinstance(file_paths, list) else file_paths
            if fname is not None:
                base = os.path.basename(str(fname)).replace('.npy', '').replace('.npz', '')
                subject_id = f"batch{batch_idx}_{base}"
            else:
                subject_id = f"batch{batch_idx}_subj{subj_idx}"

            batch_ids.append(subject_id)
            batch_spars.append(_compute_sparsification_curves(
                sampled_velocities, Sdef_series, series, subj_idx,
                error_metric=args.error_metric,
                lv_segs=lv_segs, v_series=v_series,
                inshape=(img_size, img_size),
            ))
            batch_rc.append(_compute_risk_coverage_curves(
                sampled_velocities, Sdef_series, series, subj_idx,
                error_metric=args.error_metric,
                lv_segs=lv_segs, v_series=v_series,
                inshape=(img_size, img_size),
            ))
            if args.compute_reg_metrics:
                batch_reg.append(compute_registration_metrics(
                    source_img, target_imgs, Sdef_series, v_series,
                    lv_segs, sampled_velocities, subj_idx,
                    inshape=(img_size, img_size),
                ))

        all_ids.extend(batch_ids)
        all_spars.extend(batch_spars)
        all_rc.extend(batch_rc)
        all_reg.extend(batch_reg)
        global_subj_idx += batch_size

        # ---- Per-batch report ----
        batch_folder = os.path.join(out_root, f"batch_{batch_idx:03d}")
        _write_summary(
            batch_spars, batch_rc, batch_ids, batch_folder,
            header=f"MC-EM (LANGEVIN, GD M-STEP) — batch {batch_idx}",
            extra_lines=[f"  alpha={args.alpha:g}, beta={args.beta:g}, T={T_frames}, "
                         f"N={args.em_samples}, EM rounds/sequence: "
                         f"min={int(rounds_run.min())} max={int(rounds_run.max())} "
                         f"mean={float(rounds_run.float().mean()):.1f}",
                         f"  sigma mean={float(sigma.mean()):.6f} "
                         f"std={float(sigma.std()):.6f}",
                         f"  backbone={args.backbone}, split={data_split}, "
                         f"num_runs={NUM_RUNS}, error_metric={args.error_metric}"],
        )
        if args.compute_reg_metrics:
            _write_reg_metrics(batch_reg, batch_folder)
        print(f"  per-batch results: {batch_folder}")

        del (series, lv_segs, Sdef_series, v_series, v_series_bb,
             source_img, target_imgs, scaling_factors_bb, sigma,
             v_bar_list, u_bar_list, sampled_velocities)
        torch.cuda.empty_cache()
        gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("\nReleased registration model from GPU memory")

    # ------------------------------------------------------------------
    # End-of-run aggregate
    # ------------------------------------------------------------------
    with open(os.path.join(out_root, "sigma_history.json"), "w") as f:
        json.dump(all_histories, f, indent=2)

    if args.optimize_only:
        print(f"\n--optimize_only: s^R written for {global_subj_idx} subject(s) "
              f"under {s_K_dir}")
        return

    if not all_rc:
        print("\nNothing scored — no batches processed.")
        return

    agg_folder = os.path.join(out_root, "aggregate")
    o_naurc_m, o_naurc_s, o_nause_m, o_nause_s, n_subj = _write_summary(
        all_spars, all_rc, all_ids, agg_folder,
        header="MC-EM (LANGEVIN, GD M-STEP) — ALL SUBJECTS",
        extra_lines=[f"  M-step minimizer sigma* = (2*beta + S_bar)/(2*alpha + T + 1), "
                     f"alpha={args.alpha:g}, beta={args.beta:g}, T={T_frames}",
                     f"  EM: R={args.em_rounds}, N={args.em_samples}, "
                     f"M-step Adam lr={args.gd_lr:g} iters<={args.gd_iters}",
                     f"  backbone={args.backbone}, split={data_split}, "
                     f"num_runs={NUM_RUNS}, error_metric={args.error_metric}",
                     f"  T_diff={args.num_diffusion_steps}, gamma={args.gamma}, "
                     f"guidance={args.guidance_scale}, reg_weight={args.reg_weight}"],
    )
    if args.compute_reg_metrics:
        _write_reg_metrics(all_reg, agg_folder)

    print(f"\n{'=' * 70}")
    print(f"OVERALL ({n_subj} subjects): "
          f"nAURC {o_naurc_m:.4f} +/- {o_naurc_s:.4f}   "
          f"nAUSE {o_nause_m:.4f} +/- {o_nause_s:.4f}")
    print(f"{'=' * 70}")
    print(f"\nAll results written under: {out_root}")
    print("Compare against the learned pipeline:")
    print(f"  {config.model.params.workdir}/visualization/"
          f"uncertainty_sde_combined_acdc_v3_iclr/aggregate/"
          f"per_frame_nAURC_nAUSE_summary.txt")
    print("and against the fixed-sigma ablation:")
    print("  python -m uncertainty_sde_combined_acdc_v3_iclr.main_fixed_sigma "
          f"--backbone {args.backbone} --num_runs {NUM_RUNS}")


if __name__ == "__main__":
    main()
