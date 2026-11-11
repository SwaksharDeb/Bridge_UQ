"""
Generalized EM (GEM) for an amortized sigma network sigma = f_theta(I, v_hat).

Same generative model and the same E-step as the closed-form MC-EM in
mcem.py; only the parameterization of sigma changes. Instead of a free sigma
per voxel per sequence, sigma is the output of a U-Net shared by all subjects
(SigmaRegressionNetwork: source + target frames and v_hat in, [B, 2, H, W]
positive sigma out), and EM estimates the network weights theta.

One GEM round, on a minibatch of training subjects:

  E-step  (no gradients)
      sigma_i = f_theta(I_i, v_hat_i)
      draw N chains  v ~ p(v | sigma_i, I_i)       (samplers.py: ula / sghmc /
                                                   psgld / adam_sgld)
      S_bar_i = mean over chains of S(v)           (mcem.bridge_residual_S)

  M-step  (--m_steps Adam steps on theta, targets held fixed)
      maximize  Q(theta) = sum_{i,x} -[ (2 beta + S_bar_ix) / (2 sigma_ix)
                                        + ((2 alpha + T + 1)/2) log sigma_ix ]
      i.e. minimize the Itakura-Saito divergence between sigma_theta and the
      closed-form sigma* = (2 beta + S_bar)/(2 alpha + T + 1), which is Q's
      per-voxel maximizer. Backprop runs through the U-Net only; the sampler
      is never differentiated.

A few Adam steps that INCREASE Q (rather than maximize it) is what makes this
generalized EM: the marginal likelihood is still non-decreasing (Dempster et
al. 1977; Wu 1983), approximately so with Monte Carlo E-steps. Because theta
is shared, the network regresses every subject's noisy per-voxel target onto
image features at once -- which is where the smooth, anatomy-conditioned
structure comes from. The image does not enter the objective.

--target fixed_point swaps sigma* for mcem's accelerated fixed-point target
(same fixed point, reached in fewer rounds); the default closed_form is the
plain EM map.

Outputs under --output_dir (default: the folder main.py writes to; all gem_-prefixed):
    gem_checkpoints/gem_round_NNNN.pt, gem_final.pt  network + args
    gem_history.json                              per-round loss / sigma stats
    gem_eval/sigma_vis/round_NNNN.png             sigma on held-out test batch
    gem_eval/sigma_contrast/round_NNNN.json       edge-vs-flat contrast
    gem_s_K/batch_NNN.pt (with --export_test)     test-set sigma, same keys as
                                                  main.py's s_K, for test_fast.py

Example:
    python -m uncertainty_sde_combined_acdc_v3_iclr_mcem_gd.gem_unet \
        --rounds 300 --batch_size 8 --em_samples 32 --m_steps 5 \
        --langevin_algorithm psgld --langevin_steps 100 --langevin_step_size 0.05 \
        --gamma 5e-6 --reg_weight 0
"""

import os
import sys

sys.path.append("/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2")

import argparse
import json
import time

import numpy as np
import torch

from uncertainty_sde_combined_acdc_v3_iclr.networks import SigmaRegressionNetwork
from uncertainty_sde_combined_acdc_v3_iclr_mcem_gd.mcem import (
    bridge_residual_S, sigma_init, sigma_to_scaling)
from uncertainty_sde_combined_acdc_v3_iclr_mcem_gd.samplers import (
    adaptive_langevin_velocities, ALGORITHMS)
from uncertainty_sde_combined_acdc_v3_iclr_mcem_gd.sigma_contrast import (
    edge_contrast, summarize)
from uncertainty_sde_combined_acdc_v3_iclr_mcem_gd.viz import save_sigma_panel


def q_loss(sigma, S_bar, alpha, beta, T):
    """-Q per voxel, averaged: (2 beta + S_bar)/(2 sigma) + (D/2) log sigma.

    Minimized per voxel at sigma* = (2 beta + S_bar)/D; up to a constant it is
    (D/2) * ItakuraSaito(sigma*, sigma).
    """
    D = 2.0 * alpha + T + 1.0
    return ((2.0 * beta + S_bar) / (2.0 * sigma) + 0.5 * D * torch.log(sigma)).mean()


def e_step(sigma, v_reg, src, tgt, n_samples, chunk, lkw):
    """S_bar [B, C, H, W]: mean bridge residual over n_samples chains at sigma."""
    B = sigma.shape[0]
    T = len(v_reg) - 1
    v_reg_1 = [v.unsqueeze(0) for v in v_reg]
    S_sum, drawn = None, 0
    while drawn < n_samples:
        n = min(chunk, n_samples - drawn)
        rep = lambda t: t.repeat(n, *([1] * (t.dim() - 1)))      # sample-major
        v_t, _ = adaptive_langevin_velocities(
            [rep(v) for v in v_reg], rep(sigma), rep(src), [rep(t) for t in tgt],
            **lkw)
        with torch.no_grad():
            vv = [v.detach().view(n, B, *v.shape[1:]) for v in v_t]
            S = bridge_residual_S(vv, v_reg_1, T=T).sum(0)
            S_sum = S if S_sum is None else S_sum + S
        del v_t
        drawn += n
    return S_sum / n_samples


def fixed_point_target(S_bar, sigma_old, alpha, beta, T, damping):
    """mcem's fixed_point jump, expressed as the S_bar the closed form needs."""
    D = 2.0 * alpha + T + 1.0
    ratio = (S_bar / sigma_old.clamp_min(1e-12)).clamp(max=float(T - 1))
    s_fp = 2.0 * beta / (D - ratio)
    if damping < 1.0:
        s_fp = torch.exp((1.0 - damping) * sigma_old.clamp_min(1e-12).log()
                         + damping * s_fp.log())
    return s_fp * D - 2.0 * beta


def register(model, batch, dev):
    series = batch['series'].to(dev)
    lv = batch.get('lv_segs')
    lv = lv.to(dev) if lv is not None else torch.zeros_like(series[:, 0:1])
    with torch.no_grad():
        _, v_series, _, _, _ = model.sequence_register_no_avg_lowf_addlatentf(series, lv)
    src = series[:, 0:1]
    tgt = [series[:, t:t + 1] for t in range(1, series.shape[1])]
    v_reg = [torch.zeros_like(v_series[0])] + list(v_series)
    return src, tgt, v_reg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # GEM
    ap.add_argument('--rounds', type=int, default=300,
                    help='GEM rounds; each is one E-step + M-step on a minibatch')
    ap.add_argument('--m_steps', type=int, default=5,
                    help='Adam steps on theta per round (GEM: increase Q, not maximize)')
    ap.add_argument('--lr', type=float, default=None,
                    help="Adam lr. Default None = torch.optim.Adam's default (1e-3)")
    ap.add_argument('--target', choices=['closed_form', 'fixed_point'],
                    default='closed_form',
                    help="closed_form: the EM map. fixed_point: mcem's accelerated "
                         "target with the same fixed point.")
    ap.add_argument('--em_damping', type=float, default=0.5,
                    help='--target fixed_point only: fraction of the jump, log space')
    ap.add_argument('--alpha', type=float, default=1e-4)
    ap.add_argument('--beta', type=float, default=3.0)
    ap.add_argument('--sigma_convention', choices=['code_s', 'paper_sigma'],
                    default='code_s', help='units of the saved scaling factors s')
    # data
    ap.add_argument('--backbone', default='tlrn')
    ap.add_argument('--batch_size', type=int, default=8, help='subjects per round')
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--num_diffusion_steps', type=int, default=14)
    # E-step sampler
    ap.add_argument('--em_samples', type=int, default=32, help='chains per subject')
    ap.add_argument('--sample_chunk', type=int, default=32)
    ap.add_argument('--langevin_algorithm', choices=list(ALGORITHMS), default='psgld')
    ap.add_argument('--langevin_steps', type=int, default=100)
    ap.add_argument('--langevin_step_size', type=float, default=0.05)
    ap.add_argument('--langevin_init', choices=['prior', 'zero'], default='zero')
    ap.add_argument('--langevin_precondition', choices=['none', 'diag'], default='none')
    ap.add_argument('--langevin_noise_smooth_width', type=float, default=0.0)
    ap.add_argument('--gamma', type=float, default=5e-5)
    ap.add_argument('--guidance_scale', type=float, default=1.0)
    ap.add_argument('--reg_weight', type=float, default=0.1)
    ap.add_argument('--sghmc_friction', type=float, default=0.1)
    ap.add_argument('--rms_beta', type=float, default=0.99)
    ap.add_argument('--rms_lambda', type=float, default=1.0)
    # eval / output
    ap.add_argument('--eval_every', type=int, default=5,
                    help='rounds between held-out sigma panels + checkpoints (0 = end only)')
    ap.add_argument('--no_training_vis', dest='training_vis', action='store_false',
                    default=True,
                    help="Skip the 7-row v3_iclr figure (trainer.visualize_training_"
                         "step) written at every eval to gem_training_vis/batch_000/"
                         "training_vis_iterNNNN.png.")
    ap.add_argument('--vis_subject_idx', type=int, default=None,
                    help='Test-batch subject shown in that figure. Default: one '
                         'drawn at random, kept for the whole run.')
    ap.add_argument('--vis_draws', type=int, default=8,
                    help='Posterior draws for the figure: row |v_t| shows the '
                         'first, row |std_tilde| the per-frame std across all.')
    ap.add_argument('--log_every', type=int, default=1)
    ap.add_argument('--export_test', action='store_true',
                    help='after training, write gem_s_K/batch_NNN.pt for the whole test set')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output_dir', default=None,
                    help='Default: <backbone workdir>/visualization/'
                         'uncertainty_sde_combined_acdc_v3_iclr_mcem_gd -- the folder '
                         'main.py writes to. Every GEM output is gem_-prefixed so '
                         "nothing of main.py's (run_config.json, s_K/, sigma_vis/) "
                         'is overwritten.')
    args = ap.parse_args()

    dev = 'cuda'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from torch.utils.data import DataLoader
    from uncertainty_sde_combined_acdc_v3_iclr.backbones import load_backbone_and_data
    model, train_loader, test_loader, config = load_backbone_and_data(
        backbone=args.backbone, combine_acdc=True)

    out = args.output_dir or (f"{config.model.params.workdir}/visualization/"
                              f"uncertainty_sde_combined_acdc_v3_iclr_mcem_gd")
    for d in ('gem_checkpoints', 'gem_eval/sigma_vis', 'gem_eval/sigma_contrast',
              'gem_s_K', 'gem_training_vis/batch_000'):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    with open(os.path.join(out, 'gem_run_config.json'), 'w') as f:
        json.dump(dict(vars(args), output_dir=out), f, indent=2, sort_keys=True)
    print(f"GEM outputs: {out}/gem_*")
    collate = getattr(train_loader, 'collate_fn', None)
    g = torch.Generator(); g.manual_seed(args.seed)
    train_dl = DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=True,
                          drop_last=True, num_workers=args.num_workers,
                          collate_fn=collate, generator=g)
    test_dl = DataLoader(test_loader.dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers,
                         collate_fn=getattr(test_loader, 'collate_fn', None))

    # held-out eval batch: the first test batch, the same subjects main.py shows
    ev_batch = next(iter(test_dl))
    ev_src, ev_tgt, ev_vreg = register(model, ev_batch, dev)
    T = len(ev_vreg) - 1
    H = ev_src.shape[-1]
    if T != args.num_diffusion_steps:
        raise SystemExit(f"frame count T={T} != --num_diffusion_steps "
                         f"{args.num_diffusion_steps}")

    s1 = float(sigma_init(args.alpha, args.beta, T))
    net = SigmaRegressionNetwork(num_time_steps=T, img_size=H, init_output=s1).to(dev)
    opt = (torch.optim.Adam(net.parameters()) if args.lr is None
           else torch.optim.Adam(net.parameters(), lr=args.lr))
    print(f"GEM U-Net: {sum(p.numel() for p in net.parameters()):,} params, "
          f"sigma init {s1:.4f} (= 2 beta/(2 alpha+T+1)), target={args.target}, "
          f"Adam lr={opt.param_groups[0]['lr']:g}")

    lkw = dict(algorithm=args.langevin_algorithm, gamma=args.gamma,
               guidance_scale=args.guidance_scale, reg_weight=args.reg_weight,
               inshape=(H, H), n_steps=args.langevin_steps,
               step_size=args.langevin_step_size, init=args.langevin_init,
               precondition=args.langevin_precondition,
               noise_smooth_width=args.langevin_noise_smooth_width,
               friction=args.sghmc_friction, rms_beta=args.rms_beta,
               rms_lambda=args.rms_lambda)
    print(f"E-step: {args.langevin_algorithm}, {args.em_samples} chains x "
          f"{args.langevin_steps} steps per subject, gamma={args.gamma:g}, "
          f"reg_weight={args.reg_weight:g}")

    # 7-row figure in v3_iclr's training_vis layout, drawn by the trainer's own
    # visualize_training_step (the trainer is only a host for that method and
    # warp_image; nothing is trained through it).
    Bev = ev_src.shape[0]
    vis_i = (int(np.random.randint(0, Bev)) if args.vis_subject_idx is None
             else min(max(0, args.vis_subject_idx), Bev - 1))
    vis_trainer = None
    if args.training_vis:
        from uncertainty_sde_combined_acdc_v3_iclr.trainer import ScalingFactorTrainer
        from uncertainty_sde_combined_acdc_v3_iclr_mcem_gd.mcem import (
            PerSequenceBridge, _expand_sf_bb)
        vis_trainer = ScalingFactorTrainer(
            scaling_network=None,
            brownian_bridge=PerSequenceBridge(num_diffusion_steps=T, img_size=H,
                                              reg_weight=args.reg_weight),
            loss_fn=None, device=dev, img_size=H, gamma=args.gamma,
            guidance_scale=args.guidance_scale)
        print(f"training_vis: test subject {vis_i}, {args.vis_draws} draws per figure "
              f"-> {out}/gem_training_vis/batch_000/")

    def training_vis(sig, r):
        """Draw --vis_draws chains for subject vis_i at sigma = f_theta, then
        hand batch-sized tensors (only row vis_i filled) to
        visualize_training_step so its title shows the real subject index."""
        n = max(1, int(args.vis_draws))
        sl = slice(vis_i, vis_i + 1)
        rep = lambda t: t[sl].repeat(n, *([1] * (t.dim() - 1)))
        v_draws, _ = adaptive_langevin_velocities(
            [rep(v) for v in ev_vreg], rep(sig), rep(ev_src), [rep(t) for t in ev_tgt],
            **lkw)
        with torch.no_grad():
            s_raw = sigma_to_scaling(sig, T, args.sigma_convention)
            sf_bb = _expand_sf_bb(s_raw, T + 1)[0]
            v_t_list, var_list, warped = [], [], []
            for v in v_draws:                                   # [n, C, H, W]
                v = v.detach()
                full = torch.zeros_like(ev_vreg[0]); full[vis_i] = v[0]
                var = torch.zeros_like(full)
                if n > 1:
                    var[vis_i] = v.var(dim=0)                   # row plots sqrt(var0+var1)
                v_t_list.append(full)
                var_list.append(var)
                warped.append(vis_trainer.warp_image(ev_src, full))
            vis_trainer.visualize_training_step(
                sf_bb, var_list, ev_vreg, v_t_list, ev_src, ev_tgt, warped,
                os.path.join(out, 'gem_training_vis', 'batch_000'),
                epoch=r - 1, subject_idx=vis_i)

    def evaluate(tag, r=None):
        net.eval()
        with torch.no_grad():
            sig = net(ev_src, ev_tgt, ev_vreg[1:])
        net.train()
        if vis_trainer is not None and r is not None:
            training_vis(sig, r)
        s_raw = sigma_to_scaling(sig, T, args.sigma_convention)
        save_sigma_panel(s_raw, ev_src,
                         os.path.join(out, 'gem_eval', 'sigma_vis', f'{tag}.png'),
                         per_panel=True, overlay=False,
                         title=f"GEM U-Net sigma |s| (test batch 0, {tag})")
        rows, _ = edge_contrast(s_raw, ev_src)
        summ = summarize(rows)
        with open(os.path.join(out, 'gem_eval', 'sigma_contrast', f'{tag}.json'), 'w') as f:
            json.dump({'summary': summ, 'per_subject': rows}, f, indent=2)
        return summ

    def save_ckpt(name, r):
        torch.save({'state_dict': net.state_dict(), 'round': r, 'args': vars(args),
                    'num_time_steps': T, 'img_size': H,
                    'target_field': 'sigma', 'sigma_convention': args.sigma_convention},
                   os.path.join(out, 'gem_checkpoints', name))

    history = []
    train_iter = iter(train_dl)
    t0 = time.time()
    for r in range(1, args.rounds + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)
        src, tgt, v_reg = register(model, batch, dev)

        # ---- E-step: sigma from the current network, chains, S_bar ----
        with torch.no_grad():
            sigma_old = net(src, tgt, v_reg[1:]).detach()
        S_bar = e_step(sigma_old, v_reg, src, tgt, args.em_samples,
                       args.sample_chunk, lkw)
        if args.target == 'fixed_point':
            S_bar = fixed_point_target(S_bar, sigma_old, args.alpha, args.beta, T,
                                       args.em_damping)
        S_bar = S_bar.detach()
        D = 2.0 * args.alpha + T + 1.0
        sigma_star = (2.0 * args.beta + S_bar) / D

        # ---- M-step: a few Adam steps on theta, targets fixed ----
        q_before = float(q_loss(sigma_old, S_bar, args.alpha, args.beta, T))
        for _ in range(args.m_steps):
            sigma = net(src, tgt, v_reg[1:]).clamp_min(1e-6)
            loss = q_loss(sigma, S_bar, args.alpha, args.beta, T)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            sigma_new = net(src, tgt, v_reg[1:])
            q_after = float(q_loss(sigma_new, S_bar, args.alpha, args.beta, T))

        rec = dict(round=r, negQ_before=q_before, negQ_after=q_after,
                   sigma_mean=float(sigma_new.mean()), sigma_min=float(sigma_new.min()),
                   sigma_max=float(sigma_new.max()),
                   target_mean=float(sigma_star.mean()),
                   target_std=float(sigma_star.std()),
                   seconds=time.time() - t0)
        if args.eval_every and (r % args.eval_every == 0 or r == 1):
            summ = evaluate(f'round_{r:04d}', r)
            rec['eval_sep'], rec['eval_ratio'] = summ['sep'][0], summ['ratio'][0]
            save_ckpt(f'gem_round_{r:04d}.pt', r)
        history.append(rec)
        if r % args.log_every == 0 or r == 1:
            print(f"round {r:4d}  -Q {q_before:.4f} -> {q_after:.4f}  "
                  f"sigma mean {rec['sigma_mean']:.3f} [{rec['sigma_min']:.3f}, "
                  f"{rec['sigma_max']:.3f}]  target mean {rec['target_mean']:.3f} "
                  f"std {rec['target_std']:.3f}"
                  + (f"  | eval sep {rec['eval_sep']:.3f} ratio {rec['eval_ratio']:.3f}"
                     if 'eval_sep' in rec else "")
                  + f"  ({rec['seconds']:.0f}s)", flush=True)
        with open(os.path.join(out, 'gem_history.json'), 'w') as f:
            json.dump(history, f, indent=1)
        del src, tgt, v_reg, sigma_old, S_bar, sigma_star, sigma_new

    summ = evaluate('final', args.rounds)
    save_ckpt('gem_final.pt', args.rounds)
    print("final eval (test batch 0): " + ", ".join(
        f"{k} {v[0]:.3f}" for k, v in summ.items()))

    if args.export_test:
        # Same key set as main.py's s_K artifacts so test_fast.py reads them.
        net.eval()
        for bi, batch in enumerate(test_dl):
            src, tgt, v_reg = register(model, batch, dev)
            with torch.no_grad():
                sig = net(src, tgt, v_reg[1:])
            s_raw = sigma_to_scaling(sig, T, args.sigma_convention)
            paths = batch.get('path', [None] * src.shape[0])
            torch.save({
                'batch_idx': bi,
                'subject_paths': [str(p) if p is not None else
                                  f"subj_{bi * args.batch_size + i:05d}"
                                  for i, p in enumerate(paths)],
                'scaling_factors_raw': s_raw.cpu(),
                'sigma': sig.cpu(),
                'gamma': args.gamma, 'guidance_scale': args.guidance_scale,
                'num_diffusion_steps': T, 'num_cardiac_frames': T + 1,
                'alpha': args.alpha, 'beta': args.beta,
                'sigma_convention': args.sigma_convention,
                'inner_iterations': args.rounds, 'loss_dict': None,
                'checkpoint': os.path.join(out, 'gem_checkpoints', 'gem_final.pt'),
                'mean_u': None, 'method': 'gem_unet',
            }, os.path.join(out, 'gem_s_K', f'batch_{bi:03d}.pt'))
        print(f"test-set sigma exported: {os.path.join(out, 'gem_s_K')} "
              f"({bi + 1} batches)")
    print(f"done. outputs under {out}")


if __name__ == '__main__':
    main()
