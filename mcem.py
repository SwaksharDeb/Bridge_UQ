"""
Per-sequence estimation of the BridgeUQ diffusion variance sigma.

Two M-steps: the closed-form maximizer of Eq. 10 with the samples held fixed
(`mcem_refine`, exact but blind to the image), and the total derivative taken
through the sampler (`total_derivative_refine`, which is what produces
anatomical structure). See "Why -- and the fix" below for the measured
comparison.

Replaces the amortized MAP network f_theta of
`uncertainty_sde_combined_acdc_v3_iclr` with the exact maximizer of Eq. 10.

Derivation
----------
sigma^i is diagonal and spatially independent, so Eq. 10 decouples over voxels
and dimensions. Fix a voxel x and dimension j and write sigma = sigma^i_j(x).
Let r_t = v_t - mu_{t|t-1} be the innovation at frame t, and recall from
Proposition 1(iii)

    mu_{t|t-1} = v_hat_t + rho_t * (v_{t-1} - v_hat_{t-1}),
    delta_{t|t-1} = sigma * rho_t,          rho_t = (T - t) / (T - (t - 1)).

Collecting the sigma-dependent terms of Eq. 10 (the energy term does not
involve sigma, and the t = T transition is degenerate because v_T is pinned, so
delta_{T|T-1} = 0):

    L(sigma) = -S(v)/(2 sigma) - ((T-1)/2) log sigma
               - beta/sigma - (alpha+1) log sigma + const,

    S(v) = sum_{t=1}^{T-1} ((T - (t-1)) / (T - t)) * r_t^2
         = sum_{t=1}^{T-1} r_t^2 / rho_t.

The log-determinant contributes -(1/2) sum_t log(sigma * rho_t), whose rho_t
factors are sigma-free and fold into the constant. Setting dL/dsigma = 0,

    (S/2 + beta) / sigma^2 = ((T-1)/2 + alpha + 1) / sigma
    =>  sigma* = (2 beta + S(v)) / (2 alpha + T + 1),                    (*)

which is the mode of the conjugate posterior
sigma | v ~ InvGamma(alpha + (T-1)/2, beta + S(v)/2) and hence the GLOBAL
maximizer of L -- not a stationary point that needs gradient ascent.

Given N posterior samples {v^(n)} drawn with the current sigma^r, the Monte
Carlo M-step replaces S(v) by its sample average S_bar, giving

    sigma^{r+1} = (2 beta + S_bar) / (2 alpha + T + 1).

At initialization v^(0) = v_hat every innovation vanishes (eta_t = 0 for all t,
hence mu_{t|t-1} = v_hat_t and r_t = 0), so

    sigma^1 = 2 beta / (2 alpha + T + 1)

is spatially CONSTANT; spatial structure enters only through the innovations of
subsequent samples. The loop is

    sigma^1 -> v^(n) ~ p(v | sigma^1, I, v_hat) -> sigma^2
            -> v^(n) ~ p(v | sigma^2, I, v_hat) -> ...

with the sampling step being Eq. 14 (the Euler-Maruyama discretization of the
conditional reverse SDE), i.e. exactly
`BrownianBridgeLearnedScaling.run_posterior_reverse_process`.

Units
-----
The repository carries two incompatible readings of the bridge variance:

    losses.compute_loss  (bridge_norm, losses.py:581-595)  delta = s * rho_t
    run_posterior_reverse_process (brownian_bridge.py:414) sigma_sq = 2*s/T

and its inv_gamma_prior (losses.py:605-609) is applied to `s` directly, i.e.
the FIRST reading. Since that is the convention the learned
ScalingFactorNetwork was trained under -- and the one whose magnitudes match
the reference visualization, where sqrt(s0^2+s1^2) spans ~0.4-1.5 -- the
closed form solves for `s` directly by default ('code_s'). See
`convention_scale` for the alternative ('paper_sigma') and the exact factor.

Convergence
-----------
Write the effective slope of the EM map as

    slope = kappa * (T-1) / (2 alpha + T + 1),   kappa = E[S] / ((T-1) * s),

so the loop contracts iff slope < 1, to the fixed point
2 beta / [(2 alpha + T + 1) (1 - slope)].

kappa is NOT 1, because the Euler-Maruyama sampler is not consistent with the
Prop. 1(iii) transition the M-step assumes: Eq. 14 injects Var = sigma at every
step (`sigma_z = sqrt(sigma_sq) * z`) whereas the exact reverse bridge step
needs Var = sigma * (t-1)/t. Measured with guidance_scale=0, at sigma = 0.4,
1.0 and 3.0, kappa = 1.2454 in all three cases -- scale-invariant, as a pure
discretization factor must be -- against a predicted mean over t=2..T of
t/(t-1) = 1.2446 (0.06% agreement).

Consequences:

  'paper_sigma'  the M-step and sampler agree on the model, so kappa = 1.2454
                 lands directly in the slope: 1.079 > 1. No fixed point exists
                 and sigma grows without bound (observed: 0.83, 1.30, 1.81,
                 2.37, 3.01, ... -> inf), independently of gamma -- a 100x
                 gamma sweep produced identical trajectories.
  'code_s'       the M-step expects variance s*rho_t while the sampler delivers
                 (2s/T)*rho_t, so kappa = 1.2454 * 2/T = 0.178 and the slope is
                 0.154. The loop contracts to s ~ 0.47.

Dynamic range -- the main limitation of this module
---------------------------------------------------
sigma sees anatomy ONLY through S_bar, so its whole reachable interval is

    floor   (data explains everything, S_bar -> 0)  2 beta / (2 alpha + T + 1)
    ceiling (no data signal at all)                 floor / (1 - slope)

giving a maximum achievable contrast of 1/(1 - slope), independent of beta
(which rescales both ends equally) and of gamma and N. Under 'code_s' that is

    floor 0.400,  ceiling 0.473,  max contrast 1/(1 - 0.154) = 1.18x.

Measured contrast is 1.08-1.17x, i.e. the loop already saturates essentially
the entire interval available to it. For comparison the learned
ScalingFactorNetwork spans ~0.4-1.5 (contrast ~3.7x) with smooth anatomical
structure. Sweeps confirm the bound is not a tuning artefact:

    beta  3 -> 0.01  (300x): contrast 1.12x at EVERY beta; only the magnitude
                             moves (0.669 -> 0.002, exactly proportional)
    gamma 6.4e-3 -> 3e-5 (200x): contrast 1.08x -> 1.17x

And the spatial pattern that sigma does show is Monte Carlo noise, not anatomy.
Two independent 2000-round runs (N=8, sigma averaged over the 1900 post-burn-in
rounds, ~15200 effective samples each) give:

    sigma mean          0.472863  and  0.472863   (identical to 6 decimals)
    |s| range           0.6675 - 0.6699
    spatial std / mean  0.0004
    contrast            1.003x
    split-half corr     0.0009

so at high sampling the map is flat to 0.04% and reproduces nothing. Short runs
are misleading here: the same split-half correlation measured at low sample
counts reads 0.118 (N=8), 0.368 (N=32), 0.685 (N=128), which looks like
structure emerging but is correlated speckle -- ~120x more sampling drives it
to zero rather than to one. Anything read off a low-N sigma map is noise.

So with Eq. 14 held fixed and S taken from Prop. 1(iii) as Eq. 10 specifies,
the closed-form M-step yields a spatially FLAT sigma and nAUSE ~ 1.0 (ranking
no better than random), not the learned network's anatomical map. sigma
converges in ~3 rounds; the remaining rounds buy Monte Carlo precision only,
and what they reveal is the absence of signal.

Why -- and the fix
------------------
The closed form is not wrong, it answers a different question. Eq. 10 depends
on sigma twice: explicitly through the transition and prior terms, and
IMPLICITLY through the samples v(sigma, z), which Eq. 14 generates with sigma
inside both the drift and the injected sqrt(sigma)*z. Solving
argmax_sigma p(sigma | v^r, I, v_hat) with the samples held fixed
differentiates only the explicit sigma; the energy term then carries no sigma
and drops out entirely. That is exactly why the result is blind to the image.

Taking the total derivative instead,

    dL/dsigma = (dL/dsigma)|_{v fixed} + < dL/dv , dv(sigma,z)/dsigma >,

restores the missing path: dv/dsigma propagates the injected noise and dE/dv
measures how much it degraded alignment -- large along edges, ~0 in homogeneous
regions, behaving as ~sigma*h(x)/gamma with h the local curvature of E. This is
`total_derivative_refine` below, and it needs no change to the sampler.
Measured on the same subjects, 300 steps, N=8:

                        closed form     total derivative     reference
    |s| range           0.668-0.670     0.046-3.592          ~0.4-1.5
    spatial std/mean    0.0004          0.3478
    contrast p99/p1     1.003x          7.443x               ~3.7x
    split-half corr     0.0009          0.9967
    nAUSE               ~1.02           0.961                0.545 (paper)

The split-half correlation of 0.997 across independent noise seeds is the
decisive number: the structure is reproducible, so it is anatomy rather than
speckle. Visually sigma is low along the myocardial rim and every sharp
boundary and high over the homogeneous blood pool and background, which is the
sign the paper reports.

gamma matters more in this mode than in the closed form, because it now plays
two roles at once -- the guidance strength inside Eq. 14 AND the energy/prior
balance in the objective. gamma=5e-5 (the v3_iclr default) gave std/mean 0.348
and contrast 7.4x; gamma=6.4e-3 gave std/mean 0.089 over a narrower range. Tune
gamma for this mode rather than inheriting the per-sequence guidance default.
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
from tqdm import tqdm

from uncertainty_sde_combined_acdc_v3_iclr.brownian_bridge import BrownianBridgeLearnedScaling
from uncertainty_sde_combined_acdc_v3_iclr.test_fast import _assemble_scaling_factors_bb
from uncertainty_sde_combined_acdc_v3_iclr_mcem_gd.langevin import draw_velocities
from uncertainty_sde_combined_acdc_v3_iclr.losses import Grad as LTMAGrad

# Shared l2 gradient penalty for smooth_weight, as in
# uncertainty_sde_acdc_combined_iclr_2/main_langevin.py. Stateless, so one
# module-level instance is enough; it is the same class LTMALoss uses for the
# velocity regulariser.
_SIGMA_GRAD = LTMAGrad(penalty='l2')


# Same range the learned network's output is clamped to (trainer.py:251), so the
# reverse SDE sees numerically equivalent scaling factors on both paths.
S_MIN, S_MAX = 1e-4, 1e4


class PerSequenceBridge(BrownianBridgeLearnedScaling):
    """
    BrownianBridgeLearnedScaling with a PER-SEQUENCE likelihood score.

    Why this subclass exists. The inherited `compute_energy_gradient` builds

        energy = F.mse_loss(warped, target_all) * S

    where `warped` is [S*B_eff, 1, H, W]. `F.mse_loss` reduces with mean over
    EVERY element, so it divides by S*B_eff*H*W, and the gradient that reaches
    row b is that row's own contribution divided by B_eff. Measured, with one
    sequence replicated B times so every row is identical:

        B_eff    mean|grad_E|      ratio vs B_eff=1
            1    9.054487e-05                 1.00x
            2    4.527243e-05                 2.00x
            8    1.131811e-05                 8.00x
           32    2.829527e-06                32.00x

    i.e. grad_E is proportional to 1/B_eff, exactly. Since
    B_eff = sample_chunk * batch_size, the Eq. 13 likelihood term a given
    sequence feels depends on how many OTHER sequences happen to share the GPU
    batch, and differs between the EM loop (tiled, B_eff = n_c*B) and the final
    UQ sampling (untiled, B_eff = B).

    Eq. 11/13 define the score of a single sequence's posterior
    p(v^i | I^i, v_hat^i, sigma^i); the joint of Eq. 9 is a PRODUCT over i, so
    no 1/N over sequences belongs there. Multiplying by B_eff undoes the
    spurious division and leaves each row with the gradient it would have had
    if that sequence had been run alone -- which is what makes the optimization
    genuinely per-sequence, and makes batching a pure throughput device with no
    effect on the result.

    Consequence for gamma. The correction enters Eq. 14 as
    (guidance_scale/gamma) * sigma_sq * grad_E, so removing a 1/B_eff makes the
    guidance B_eff times stronger than the inherited path at the same gamma.
    The gamma=5e-5 default was tuned against the shrunken gradient, so a
    per-sequence run needs gamma scaled up by roughly the B_eff it was tuned at
    (see --guidance_norm / --gamma in main.py).
    """

    def compute_energy_gradient(self, v_t, v_reg_list, source_img, target_imgs,
                                paper_t, num_cardiac_frames):
        g = super().compute_energy_gradient(
            v_t, v_reg_list, source_img, target_imgs, paper_t, num_cardiac_frames)
        # v_t.shape[0] == B_eff == the exact factor mse_loss divided in by.
        return g * float(v_t.shape[0])


def bridge_ratio(t, T):
    """rho_t = (T - t) / (T - (t - 1)), the Proposition 1(iii) variance ratio."""
    return (T - t) / (T - (t - 1))


def bridge_residual_S(v_t_list, v_reg_list, T=None):
    """
    S(v) = sum_{t=1}^{T-1} r_t^2 / rho_t, per voxel and per channel.

    Args:
        v_t_list:   list of T+1 tensors [B, C, H, W] -- one sampled trajectory,
                    indexed by cardiac frame 0..T (both ends boundary-pinned).
        v_reg_list: list of T+1 tensors [B, C, H, W] -- the pretrained v_hat.
        T:          frame count; defaults to len(v_t_list) - 1.

    Returns:
        [B, C, H, W] fp32 tensor. Exactly zero when v_t_list == v_reg_list,
        which is the sigma^1 initialization case.

    The innovation is computed as r_t = eta_t - rho_t * eta_{t-1} with
    eta_t = v_t - v_hat_t, which is algebraically identical to
    v_t - mu_{t|t-1} but keeps both terms small (the v_hat part cancels
    analytically instead of numerically).

    t = T is excluded: delta_{T|T-1} = sigma * rho_T = 0 there, a degenerate
    transition carrying no information about sigma.
    """
    if T is None:
        T = len(v_t_list) - 1
    if T < 2:
        raise ValueError(f"Need at least 2 frames to form an innovation, got T={T}")

    with torch.no_grad():
        eta_prev = (v_t_list[0].detach().float()
                    - v_reg_list[0].detach().float())
        S = torch.zeros_like(eta_prev)
        for t in range(1, T):
            eta_t = v_t_list[t].detach().float() - v_reg_list[t].detach().float()
            rho_t = bridge_ratio(t, T)
            r_t = eta_t - rho_t * eta_prev
            S = S + (r_t * r_t) / rho_t
            eta_prev = eta_t
    return S


def sigma_star(S_bar, alpha, beta, T):
    """
    Closed-form M-step, Eq. (*): sigma* = (2 beta + S_bar) / (2 alpha + T + 1).

    S_bar may be a tensor (the Monte-Carlo average of bridge_residual_S over N
    trajectories) or a float. The denominator is a positive constant, so sigma*
    inherits S_bar's positivity and needs no clamping of its own.
    """
    return (2.0 * beta + S_bar) / (2.0 * alpha + T + 1.0)


def sigma_star_smooth(S_bar, alpha, beta, T, weight=0.0, iters=200, lr=0.05):
    """
    Closed-form M-step with an l2 SPATIAL SMOOTHNESS penalty on sigma.

    Minimizes, per sequence, the negative log of the conjugate posterior plus
    the same penalty total_derivative_refine uses:

        F(sigma) = mean_x[ (S_bar/2 + beta)/sigma + c*log sigma ]
                   + weight * Grad_l2(sigma),      c = (T-1)/2 + alpha + 1.

    With weight = 0 the minimizer is sigma_star exactly (dF/dsigma = 0 gives
    (2 beta + S_bar)/(2 alpha + T + 1)), and that is returned without
    iterating. With weight > 0 the penalty couples neighbouring voxels, so
    there is no closed form any more: Adam on u = log sigma (keeps sigma > 0),
    started from sigma_star. Both terms are means over the same elements, so
    the problem stays independent per sequence and per channel.

    SCALE: the data term's curvature per voxel is ~c/sigma^2, the penalty's
    ~4*weight, so the smoothing length is roughly sqrt(weight*sigma^2/c)
    pixels -- with c = 7.5 at T = 14 and sigma ~ 1, weight 0.05 barely
    smooths; ~10-100 is needed for a few pixels. This is a different scale
    from the total-derivative loss, where the same weight competes with the
    gamma-scaled LTMALoss.
    """
    sigma0 = sigma_star(S_bar, alpha, beta, T)
    if weight is None or weight <= 0:
        return sigma0
    a = (0.5 * S_bar + beta).detach().float()
    c = 0.5 * (T - 1) + alpha + 1.0
    u = sigma0.detach().float().clamp_min(1e-12).log().clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    with torch.enable_grad():
        for _ in range(int(iters)):
            opt.zero_grad(set_to_none=True)
            sig = u.exp()
            F_ = (a / sig + c * u).mean() + weight * _SIGMA_GRAD.loss(sig)
            F_.backward()
            opt.step()
    return u.detach().exp().to(sigma0.dtype)


def sigma_star_gd(S_bar, alpha, beta, T, sigma_init_tensor, iters=500, lr=0.05,
                  tol=1e-7, weight=0.0, optimizer='gd', max_step=1.0):
    """
    GRADIENT-DESCENT M-step: the same maximization sigma_star solves in closed
    form, done iteratively and WITHOUT the explicit solution.

    With the samples held fixed, the sigma-dependent part of the negative
    expected complete-data log posterior is, per voxel and channel,

        F(sigma) = (S_bar/2 + beta)/sigma + c*log sigma,   c = (T-1)/2 + alpha + 1,

    exactly the objective whose stationary point dF/dsigma = 0 gives
    sigma* = (2 beta + S_bar)/(2 alpha + T + 1). Here sigma* is never used to
    set sigma: F is minimized iteratively on u = log sigma (keeps sigma > 0;
    F(u) = a e^{-u} + c u is convex in u, F'' = a e^{-u} > 0), starting from
    `sigma_init_tensor` -- normally the previous round's sigma -- and stopping
    once the per-voxel gradient |dF/du| = |c - a e^{-u}| is below `tol`
    everywhere, or after `iters` steps.

    optimizer='gd' (default): plain fixed-step gradient descent,
        u <- u - (1/c) * dF/du,  with each step clipped to |du| <= max_step.
      1/c is the inverse of F's curvature AT the optimum (F'' = a e^{-u*} = c),
      so it is the natural, data-independent step: far from the optimum the
      clip keeps sigma from jumping more than a factor e^max_step per
      iteration, near it the iteration is asymptotically Newton and converges
      quadratically. Reaches the fp32 floor in a few dozen iterations even
      when a voxel's target moves by 400x in one round (which happens).
    optimizer='adam': torch Adam on u at learning rate `lr`. Converges when the
      target is within ~10x of the start, but STALLS on large moves: Adam's
      second moment (beta2 = 0.999) remembers the huge initial gradient
      a e^{-u0}, so the step m/sqrt(v) collapses as the gradient shrinks
      along the way -- measured on real data: stuck at 0.99 relative
      deviation after 500 iterations for voxels whose target was 445x the
      start. Kept as an option, not the default.

    The data term is SUMMED over voxels, not averaged, so the gradient is the
    exact per-voxel dF/du (O(1) in scale) and the stopping test does not
    depend on the tensor size; the smoothness penalty (same _SIGMA_GRAD term
    as sigma_star_smooth) is multiplied by numel to keep its relative weight
    identical to sigma_star_smooth's mean-form objective. With weight = 0 the
    voxels are independent and the minimizer is sigma* -- which is the check
    `info['rel_dev_vs_closed_form']` reports (max |sigma - sigma*| / sigma*;
    diagnostic only). The solve runs in float64 so the only error left is the
    fp32 rounding of S_bar and of the returned sigma.

    Returns (sigma, info) with info = {iters_run, grad_max, rel_dev_vs_closed_form}.
    """
    a = (0.5 * S_bar + beta).detach().double()
    c = 0.5 * (T - 1) + alpha + 1.0
    numel = float(a.numel())
    u = (sigma_init_tensor.detach().double().clamp_min(1e-300).log()
         .clone().requires_grad_(True))
    use_pen = weight is not None and weight > 0
    opt = torch.optim.Adam([u], lr=lr) if optimizer == 'adam' else None
    if optimizer not in ('gd', 'adam'):
        raise ValueError(f"Unknown optimizer: {optimizer!r}")
    iters_run, grad_max = 0, float('inf')
    with torch.enable_grad():
        for it in range(int(iters)):
            iters_run = it
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            elif u.grad is not None:
                u.grad = None
            if use_pen:
                sig = u.exp()
                F_ = (a / sig + c * u).sum() + weight * numel * _SIGMA_GRAD.loss(sig)
                F_.backward()
                g = u.grad
            else:
                # Same gradient autograd would return for sum(a e^{-u} + c u),
                # written out: no graph needed when voxels are independent.
                with torch.no_grad():
                    g = c - a * torch.exp(-u)
            grad_max = float(g.abs().max())
            if grad_max < tol:
                break
            with torch.no_grad():
                if opt is not None:
                    u.grad = g if use_pen else g.clone()
                    opt.step()
                else:
                    u -= (g / c).clamp(-max_step, max_step)
    sigma = u.detach().exp()
    with torch.no_grad():
        sigma0 = sigma_star(S_bar, alpha, beta, T).double()
        rel_dev = float(((sigma - sigma0).abs() / sigma0.clamp_min(1e-300)).max())
    info = {'iters_run': int(iters_run), 'grad_max': grad_max,
            'rel_dev_vs_closed_form': rel_dev}
    return sigma.to(S_bar.dtype), info


def sigma_init(alpha, beta, T):
    """sigma^1 = 2 beta / (2 alpha + T + 1) -- the S_bar = 0 case, a scalar."""
    return 2.0 * beta / (2.0 * alpha + T + 1.0)


def convention_scale(convention, T_diff):
    """
    Factor converting the quantity the closed form solves for into the
    `scaling_factors` s that the reverse SDE consumes.

    The repository carries TWO incompatible readings of the bridge variance,
    and the closed form has to commit to one:

      'code_s'      delta_{t|t-1} = s * rho_t. This is what
                    `losses.compute_loss` uses for bridge_norm
                    (losses.py:581-595) and what its inv_gamma_prior is applied
                    to (losses.py:605-609) -- i.e. the convention the learned
                    ScalingFactorNetwork was actually trained under. The closed
                    form then solves directly for s, and the factor is 1.
                    At alpha=1e-4, beta=3, T=14 this gives s^1 = 0.400, and the
                    loop settles near 0.47, matching the range the learned
                    network produces (~0.4-1.5 for sqrt(s0^2+s1^2)).

      'paper_sigma' delta_{t|t-1} = sigma * rho_t with sigma the paper's
                    diffusion variance, which `run_posterior_reverse_process`
                    relates to s by sigma_sq = 2*s/T (brownian_bridge.py:414).
                    The factor is then T_diff/2, giving s^1 = 2.80.

    'code_s' is the default because it is the one the reference visualization
    and the trained network agree on. Note also that 'paper_sigma' does not
    converge at these hyperparameters: the Euler-Maruyama sampler injects
    Var = sigma per step where the exact reverse bridge step needs
    Var = sigma*(t-1)/t, inflating E[S] by a measured kappa = 1.2454 (predicted
    mean(t/(t-1)) = 1.2446), which puts the EM slope
    kappa*(T-1)/(2*alpha+T+1) = 1.079 above 1 -- no fixed point. Under
    'code_s' the same sampler feeds a model expecting variance s*rho_t while
    delivering (2s/T)*rho_t, so the effective slope is 0.154 and the loop
    contracts.
    """
    if convention == 'code_s':
        return 1.0
    if convention == 'paper_sigma':
        return T_diff / 2.0
    raise ValueError(f"Unknown convention: {convention!r}")


def sigma_to_scaling(sigma, T_diff, convention='code_s'):
    """Closed-form variable -> `scaling_factors` s, clamped to the SDE's range."""
    return torch.clamp(sigma * convention_scale(convention, T_diff),
                       min=S_MIN, max=S_MAX)


def scaling_to_sigma(s, T_diff, convention='code_s'):
    """Inverse of sigma_to_scaling (before clamping)."""
    return s / convention_scale(convention, T_diff)


def fill_border(x, border):
    """Overwrite the outer `border` pixels of [..., H, W] with the nearest
    interior pixel (replicate padding of the interior). border <= 0: unchanged."""
    b = int(border)
    if b <= 0:
        return x
    return torch.nn.functional.pad(x[..., b:-b, b:-b], (b, b, b, b), mode='replicate')


def mcem_refine(trainer, source_img, target_imgs, v_reg_list,
                rounds=20, num_samples=32, sample_chunk=0,
                alpha=1e-4, beta=3.0, tol=1e-3, convention='code_s',
                average_last=0, average_after=None, log_every=1,
                sampler='reverse_sde', langevin_kw=None,
                round_callback=None,
                smooth_weight=0.0, smooth_iters=200, smooth_lr=0.05,
                chain_kw=None, em_update='closed_form', em_damping=0.5,
                m_solver='closed_form', gd_kw=None, border=0,
                device='cuda', progress=True, verbose=True):
    """
    The MC-EM loop: sigma^1 -> sample -> sigma^2 -> sample -> ...

    m_solver selects HOW the M-step maximization is carried out, not WHAT it
    maximizes: 'closed_form' (default) evaluates sigma* = (2 beta + S_bar) /
    (2 alpha + T + 1) directly (sigma_star / sigma_star_smooth);
    'gradient_descent' minimizes the same fixed-sample objective
    F(sigma) = (S_bar/2 + beta)/sigma + c log sigma by Adam on log sigma
    (sigma_star_gd), warm-started from the previous round's sigma (or from
    the constant sigma^1 if gd_kw['init'] == 'prior'), and never evaluates the
    explicit solution. Round 1 is sigma^1 under both, since S = 0 there. The
    two must agree to solver tolerance; each round's record carries
    'gd_rel_dev_vs_closed_form' (max |sigma_gd - sigma*|/sigma*) to show it.
    gd_kw = {iters, lr, tol, init}.

    Each round is ordered SOLVE-THEN-SAMPLE, matching the loop as stated:

        sigma^1 = 2 beta/(2 alpha + T + 1)          [S = 0, since v^(0) = v_hat]
          -> v^(n) ~ p(v | sigma^1, I, v_hat)       [Eq. 14]
          -> sigma^2 = (2 beta + S_bar)/(2 alpha + T + 1)
          -> v^(n) ~ p(v | sigma^2, I, v_hat)  -> ...

    i.e. the closed form runs at the TOP of the round and the draws are made
    with the sigma it just produced; S_bar from those draws feeds the next
    round's M-step. Consequently the returned sigma is the one the returned
    v_bar_list / u_list were actually sampled under. No network, no optimizer,
    no gradients w.r.t. any parameter.

    PER SEQUENCE. Eq. 9 factorizes over subjects, so each image sequence has
    its own sigma^i and its own EM problem. Everything here is elementwise in
    the batch axis -- sigma is [B, C, H, W], and S sums only over the sample
    axis -- so the M-step is already independent per sequence. Two further
    things make the *schedule* per sequence as well:

      * the convergence test is per sequence: `rel_change` is a [B] vector, and
        a sequence freezes its own sigma as soon as ITS relative change drops
        below `tol`, instead of the whole batch stopping on a batch-mean;
      * a frozen sequence is dropped from the batch, so later rounds sample
        only the sequences still moving.

    Dropping rows changes B_eff, which would silently rescale the Eq. 13
    likelihood score under the inherited bridge -- with `PerSequenceBridge`
    (used by main.py) it does not, which is what makes the subsetting sound.
    Run with the inherited bridge and results become batch-composition
    dependent; mcem_refine warns once if it sees that.

    Not wrapped in torch.no_grad(): `compute_energy_gradient` calls
    torch.autograd.grad on the warp, which needs grad mode enabled. Everything
    it returns is detached, so nothing accumulates -- same caveat as
    `trainer.inference_refine` and main.py's posterior sampling loop.

    sample_chunk > 0 draws the N trajectories `sample_chunk` at a time,
    accumulating running sums instead of holding all N, so peak memory grows as
    sample_chunk*B rather than N*B.

    Args:
        trainer:    ScalingFactorTrainer built with scaling_network=None.
        v_reg_list: [v_hat_0 (zeros), ..., v_hat_T], length T+1.
        rounds:     max EM rounds R, per sequence.
        tol:        a sequence stops once its own mean|sigma^{r+1} - sigma^r| /
                    mean(sigma^r) drops below this. Set <= 0 to run every
                    sequence for the full `rounds`.
                    NOTE the round-to-round change cannot fall below the Monte
                    Carlo noise floor of S_bar, which scales as ~1/sqrt(N):
                    measured 2.3e-2 at N=8. A tol below that never fires and the
                    loop just runs every round. Either raise N, set tol above
                    the floor, or (preferred) leave tol=0 and use average_last.
        em_update:  'closed_form' (default) is the EM map
                    sigma <- (2 beta + S_bar)/(2 alpha + T + 1). Where the
                    samples spread like the prior (kappa = S_bar/((T-1) sigma)
                    ~ 1, flat regions) its slope is ~(T-1)/(2 alpha+T+1) =
                    0.87, so it closes only ~13% of the gap per round and
                    needs ~30 rounds. 'fixed_point' treats this round's kappa
                    as fixed and jumps to the map's fixed point for it,
                        sigma <- 2 beta / (2 alpha + T + 1 - S_bar/sigma_old),
                    which has the SAME fixed point (rearrange:
                    sigma (2a+T+1) - S_bar = 2 beta) but reaches it in a few
                    rounds. kappa is capped at 1 (S_bar/sigma_old <= T-1): a
                    chain cannot legitimately spread more than the prior, and
                    kappa > 1 -- ULA bias or Monte Carlo noise -- is exactly
                    where the jump (and the slow map, eventually) runs away.
                    The cap bounds sigma by the prior-only value
                    beta/(alpha+1). The jump is fed to the M-step as the
                    equivalent S_bar, so smooth_weight still applies.
        em_damping: fixed_point only. Fraction lambda of the jump taken, in
                    log space: log sigma <- (1-lambda) log sigma_old
                    + lambda log sigma_fp. 1 = the full jump. The jump assumes
                    kappa was measured by chains equilibrated at sigma_old;
                    with warm-started chains and few steps per round they lag,
                    and the full jump oscillates -- measured on the real batch
                    (50 steps/round): mean |s| 0.57 -> 3.81 -> 1.52 -> 4.18,
                    pinned at the cap with contrast collapsing to 1.27x. After
                    a jump down the chains are still spread for the larger
                    sigma (kappa reads > 1, capped, jump to the max); after a
                    jump up they are still tight (kappa low, jump down).
                    Default 0.5.
        chain_kw:   sampler='langevin' only. None keeps one sample per chain:
                    the final state after langevin_kw['n_steps'] steps from
                    langevin_kw['init'], restarted every round. A dict
                    {burn_in, thin, n_samples, warm_start, reburn} instead runs
                    burn_in discarded steps, then collects n_samples states
                    `thin` steps apart (one langevin_velocities call with
                    collect_every=thin), so each chain contributes n_samples
                    draws to S_bar and the moments -- N_eff = N * n_samples.
                    warm_start=True keeps every chain across EM rounds: round 1
                    pays burn_in, later rounds resume from where the chain
                    stopped and run only `reburn` steps before collecting.
                    Chains are stored per sample chunk for the FULL batch and
                    subset with the active sequences, so tol > 0 is supported.
        smooth_weight: l2 spatial smoothness penalty on sigma in the M-step
                    (see sigma_star_smooth). 0 = the exact closed form. Round 1
                    is constant, so it is unaffected. smooth_iters / smooth_lr
                    are that solver's Adam iterations and step size.
        average_last: average sigma over the final k rounds before returning.
                    Once the iteration reaches its fixed point (3 rounds at the
                    default settings) every later round is the same value plus
                    independent MC noise, so averaging k of them cuts that noise
                    by sqrt(k) at no extra cost. 0/1 disables.

    Returns:
        scaling_factors_bb: [B, T+1, C, H, W] -- s from the final sigma, the
                            tensor the reverse SDE consumes.
        sigma:              [B, C, H, W] -- final sigma in PAPER units.
        v_bar_list:         list of T+1 [B, C, H, W], each sequence's sample
                            mean from the last round it was active in.
        u_list:             list of T+1 [B, C, H, W], likewise the sample std
                            (Algorithm 1 line 14 uncertainty).
        history:            list of per-round dicts.
        rounds_run:         [B] int tensor -- EM rounds each sequence used.
    """
    bb = trainer.brownian_bridge
    num_cardiac_frames = len(v_reg_list)
    T = num_cardiac_frames - 1
    N = max(1, int(num_samples))
    B = v_reg_list[0].shape[0]
    chunk = int(sample_chunk) if int(sample_chunk) > 0 else N

    # Burn-in / thinning / persistent chains (chain_kw). Chain states are kept
    # per sample chunk, keyed by the chunk's offset, for the FULL batch:
    # [n_c*B, ...] sample-major, so the rows of the active sequences are
    # view(n_c, B, ...)[:, idx].
    use_chain = sampler == 'langevin' and chain_kw is not None
    chain_state = {}
    if use_chain:
        c_burn = int(chain_kw.get('burn_in', 0))
        c_thin = max(1, int(chain_kw.get('thin', 1)))
        c_ns = max(1, int(chain_kw.get('n_samples', 1)))
        c_warm = bool(chain_kw.get('warm_start', False))
        c_reburn = int(chain_kw.get('reburn', 0))

    def _chain_draw(args_c, n_c, sigma_c, key, idx, Ba, consume):
        """Burn in (or resume), then hand c_ns states, c_thin steps apart, to
        `consume` one at a time.

        STREAMED, not collected: the chain runs in segments of c_thin steps,
        each resuming from the last (init_eta), and every state is consumed and
        dropped before the next segment. Holding all c_ns states instead cost
        c_ns x (n_c*Ba chains x T+1 frames x C x H x W) -- 39 GB for 100 states
        of 800 chains -- and ran out of memory. The noise is drawn in the same
        order, so this is the same chain as one collect_every call.
        """
        kw = dict(langevin_kw or {})
        kw.pop('n_steps', None)
        prev = chain_state.get(key) if c_warm else None
        xi = None
        if prev is not None:
            xi = [x.view(n_c, B, *x.shape[1:])[:, idx].reshape(n_c * Ba, *x.shape[1:])
                  for x in prev]
        pre = c_reburn if xi is not None else c_burn
        if pre > 0:
            _, _, xi = draw_velocities(
                *args_c, num_samples=n_c, sigma=sigma_c,
                langevin_kw=dict(kw, n_steps=pre, init_eta=xi, return_latent=True))
        for _ in range(c_ns):
            seg_kw = dict(kw, n_steps=c_thin, return_latent=True)
            if xi is not None:
                seg_kw['init_eta'] = xi
            v_t, _, xi = draw_velocities(
                *args_c, num_samples=n_c, sigma=sigma_c, langevin_kw=seg_kw)
            consume(v_t)
            del v_t
        if c_warm:
            if prev is None:
                prev = [torch.zeros(n_c * B, *x.shape[1:], device=x.device,
                                    dtype=x.dtype) for x in xi]
            for full, new in zip(prev, xi):
                full.view(n_c, B, *full.shape[1:])[:, idx] = \
                    new.detach().view(n_c, Ba, *new.shape[1:])
            chain_state[key] = prev

    if not isinstance(bb, PerSequenceBridge):
        print("    [warn] bridge is not a PerSequenceBridge: the Eq. 13 score "
              "carries a 1/B_eff factor, so results depend on batch "
              "composition and are not per-sequence.")

    # sigma is produced by the M-step at the TOP of each round. S_bar carries
    # the statistic from the previous round's samples; None means "no samples
    # yet", i.e. S = 0, which the closed form maps to
    # sigma^1 = 2 beta / (2 alpha + T + 1) -- the v^(0) = v_hat case.
    sigma = torch.full_like(v_reg_list[0].detach().float(),
                            float(sigma_init(alpha, beta, T)))
    S_bar_full = None

    # Per-sequence state. `active` drops a sequence once its own sigma settles;
    # v_bar/u keep whatever that sequence produced in its last active round.
    active = torch.ones(B, dtype=torch.bool, device=sigma.device)
    rounds_run = torch.zeros(B, dtype=torch.long, device=sigma.device)
    v_bar_list = [v.detach().float().clone() for v in v_reg_list]
    u_list = [torch.zeros_like(v) for v in v_bar_list]

    history = []
    recent = []          # trailing sigma snapshots for average_last
    # Running mean of sigma over rounds > average_after. O(1) memory, which is
    # what makes very long runs (R in the thousands) practical: storing
    # `average_last` snapshots would cost k * B * C * H * W floats.
    run_sum, run_cnt = None, 0

    if verbose:
        print(f"    MC-EM per sequence: B={B}, T={T}, alpha={alpha:g}, "
              f"beta={beta:g}, N={N} (chunk {chunk}), R<={rounds}, tol={tol:g}")
        print(f"    sigma^1 = 2*beta/(2*alpha+T+1) = "
              f"{float(sigma_init(alpha, beta, T)):.6f} (constant)")
        if use_chain:
            print(f"    Langevin chain: burn-in {c_burn}, {c_ns} sample(s) every "
                  f"{c_thin} step(s) -> N_eff = {N}x{c_ns} = {N * c_ns} per round"
                  + (f"; warm start across rounds, reburn {c_reburn}"
                     if c_warm else "; chains restart every round"))

    rounds_bar = tqdm(range(int(rounds)), desc="  MC-EM rounds",
                      leave=False, position=1, disable=not progress)

    for r in rounds_bar:
        idx = torch.nonzero(active, as_tuple=False).flatten()
        if idx.numel() == 0:
            break
        Ba = int(idx.numel())

        # Subset to the sequences still moving. Sound only because
        # PerSequenceBridge makes the score independent of B_eff.
        src_a = source_img[idx]
        tgt_a = [t[idx] for t in target_imgs]
        vreg_a = [v[idx] for v in v_reg_list]

        # ---- M-step FIRST: the closed form, per sequence ----
        # sigma^r = (2 beta + S_bar(round r-1)) / (2 alpha + T + 1), and on the
        # first round S_bar is absent, so this returns sigma^1 exactly.
        gd_info = None
        with torch.no_grad():
            sigma_old_a = sigma[idx]
            if S_bar_full is None:
                sigma_new_a = torch.full_like(
                    sigma_old_a, float(sigma_init(alpha, beta, T)))
            else:
                S_in = S_bar_full[idx]
                if em_update == 'fixed_point':
                    # Jump to the fixed point for this round's kappa (see
                    # em_update), then express it as the S_bar the closed form
                    # would need to produce it, so the M-step below -- plain
                    # or smoothness-penalized -- is unchanged.
                    ratio = (S_in / sigma_old_a.clamp_min(1e-12)).clamp(max=float(T - 1))
                    denom = 2.0 * alpha + T + 1.0
                    sigma_fp = 2.0 * beta / (denom - ratio)
                    lam = float(em_damping)
                    if lam < 1.0:
                        sigma_fp = torch.exp((1.0 - lam) * sigma_old_a.clamp_min(1e-12).log()
                                             + lam * sigma_fp.log())
                    S_in = sigma_fp * denom - 2.0 * beta
                elif em_update != 'closed_form':
                    raise ValueError(f"Unknown em_update: {em_update!r}")
                if m_solver == 'gradient_descent':
                    # Same objective, solved iteratively: Adam on log sigma
                    # from the previous sigma (or the constant sigma^1). The
                    # closed form is evaluated inside only for the reported
                    # deviation, never to set sigma.
                    gk = dict(gd_kw or {})
                    init_a = (torch.full_like(sigma_old_a, float(sigma_init(alpha, beta, T)))
                              if gk.get('init', 'previous') == 'prior' else sigma_old_a)
                    sigma_new_a, gd_info = sigma_star_gd(
                        S_in, alpha, beta, T, init_a,
                        iters=gk.get('iters', 500), lr=gk.get('lr', 0.05),
                        tol=gk.get('tol', 0.0), weight=smooth_weight,
                        optimizer=gk.get('optimizer', 'gd'),
                        max_step=gk.get('max_step', 1.0))
                elif m_solver == 'closed_form':
                    # smooth_weight > 0: penalized M-step (sigma_star_smooth),
                    # iterative; 0 is the exact closed form.
                    sigma_new_a = sigma_star_smooth(
                        S_in, alpha, beta, T, weight=smooth_weight,
                        iters=smooth_iters, lr=smooth_lr)
                else:
                    raise ValueError(f"Unknown m_solver: {m_solver!r}")

            # Per-sequence relative change: reduce over (C, H, W) only.
            red = tuple(range(1, sigma_new_a.dim()))
            denom_a = sigma_old_a.abs().mean(dim=red).clamp_min(1e-12)
            rel_a = ((sigma_new_a - sigma_old_a).abs().mean(dim=red) / denom_a)

            # border > 0: the outer `border` pixels are not estimated; they
            # take the value of the nearest interior pixel, so the sampler
            # never sees the M-step's border ring.
            sigma_new_a = fill_border(sigma_new_a, border)
            sigma[idx] = sigma_new_a
            rounds_run[idx] = r + 1

        # ---- E-step: N draws from p(v | sigma^r, I, v_hat) via Eq. 14,
        # using the sigma the M-step just produced. ----
        s_raw_a = sigma_to_scaling(sigma[idx], bb.T, convention)  # [Ba, C, H, W]
        sfbb_a = _assemble_scaling_factors_bb(
            [s_raw_a[i] for i in range(Ba)], num_cardiac_frames, device=device,
        )

        # Running sums over every trajectory drawn this round (chains x states).
        acc = {'S': None, 'n': 0,
               'v1': [None] * num_cardiac_frames, 'v2': [None] * num_cardiac_frames}
        v_reg_1 = [v.detach().float().unsqueeze(0) for v in vreg_a]
        drawn = 0
        while drawn < N:
            n_c = min(chunk, N - drawn)

            def _rep(t):
                return t.repeat(n_c, *([1] * (t.dim() - 1)))

            def _consume(v_t_list_NB, n_c=n_c):
                with torch.no_grad():
                    # [n_c*Ba, C, H, W] -> [n_c, Ba, C, H, W]. The tiling is
                    # sample-major (`repeat` stacks whole copies), so axis 0
                    # indexes the trajectory and axis 1 the sequence.
                    v_chunk = [v.view(n_c, Ba, *v.shape[1:]).detach().float()
                               for v in v_t_list_NB]

                    # S(v) per trajectory, summed over the chunk. Computed on
                    # the [n_c, Ba, ...] view in one shot -- bridge_residual_S
                    # is elementwise in the leading axes.
                    S_chunk = bridge_residual_S(v_chunk, v_reg_1, T=T).sum(dim=0)
                    acc['S'] = S_chunk if acc['S'] is None else acc['S'] + S_chunk

                    # Sample moments about v_hat: the offset is constant across
                    # the draws and close to every v^(n), so the sum of squares
                    # stays small and E[x^2]-E[x]^2 below keeps its precision.
                    for fi, v_view in enumerate(v_chunk):
                        c = v_view - v_reg_1[fi]
                        s1 = c.sum(dim=0)
                        s2 = (c * c).sum(dim=0)
                        acc['v1'][fi] = s1 if acc['v1'][fi] is None else acc['v1'][fi] + s1
                        acc['v2'][fi] = s2 if acc['v2'][fi] is None else acc['v2'][fi] + s2
                    acc['n'] += n_c

            args_c = (sampler, trainer, [_rep(v) for v in vreg_a], _rep(sfbb_a),
                      _rep(src_a), [_rep(t) for t in tgt_a])
            if use_chain:
                _chain_draw(args_c, n_c, _rep(sigma[idx]), drawn, idx, Ba, _consume)
            else:
                v_t_list_NB, _ = draw_velocities(
                    *args_c, num_samples=n_c, sigma=_rep(sigma[idx]),
                    langevin_kw=langevin_kw)
                _consume(v_t_list_NB)
                del v_t_list_NB
            del args_c
            drawn += n_c
        del v_reg_1
        S_sum, v_sum, v_sumsq, n_draws = acc['S'], acc['v1'], acc['v2'], acc['n']

        # ---- Stash S_bar for the NEXT round's M-step, and record the
        # posterior moments of the draws just made at this round's sigma. ----
        with torch.no_grad():
            if S_bar_full is None:
                S_bar_full = torch.zeros_like(sigma)
            S_bar_full[idx] = S_sum / n_draws

            for fi in range(num_cardiac_frames):
                mean_c = v_sum[fi] / n_draws
                v_bar_list[fi][idx] = vreg_a[fi].detach().float() + mean_c
                u_list[fi][idx] = (v_sumsq[fi] / n_draws - mean_c ** 2
                                   ).clamp_min(0.0).sqrt()

            # A sequence that has settled freezes and leaves the batch.
            if tol > 0:
                active[idx] = rel_a >= tol
            n_active = int(active.sum().item())

            rec = {
                'round': r + 1,
                'n_active_before': Ba,
                'n_active_after': n_active,
                'sigma_mean': float(sigma.mean().item()),
                'sigma_std': float(sigma.std().item()),
                'sigma_min': float(sigma.min().item()),
                'sigma_max': float(sigma.max().item()),
                'rel_change_mean': float(rel_a.mean().item()),
                'rel_change_max': float(rel_a.max().item()),
                'm_solver': m_solver,
            }
            if gd_info is not None:
                rec['gd_iters'] = gd_info['iters_run']
                rec['gd_grad_max'] = gd_info['grad_max']
                rec['gd_rel_dev_vs_closed_form'] = gd_info['rel_dev_vs_closed_form']
        history.append(rec)

        rounds_bar.set_postfix({
            'sigma': f"{rec['sigma_mean']:.4f}",
            'd': f"{rec['rel_change_max']:.2e}",
            'act': n_active,
        })
        if verbose and (log_every <= 1 or (r + 1) % log_every == 0 or r == 0):
            print(f"    round {r + 1:>3}: {Ba:>4} active  "
                  f"sigma mean={rec['sigma_mean']:.6f} "
                  f"std={rec['sigma_std']:.6f} "
                  f"[{rec['sigma_min']:.6f}, {rec['sigma_max']:.6f}]  "
                  f"|dsigma|/sigma mean={rec['rel_change_mean']:.3e} "
                  f"max={rec['rel_change_max']:.3e}  -> {n_active} still active"
                  + (f"\n              GD M-step: {gd_info['iters_run'] + 1} iters, "
                     f"max|dF/du|={gd_info['grad_max']:.2e}, max rel dev vs "
                     f"closed form={gd_info['rel_dev_vs_closed_form']:.2e}"
                     if gd_info is not None else ""))
        if round_callback is not None:
            # s the draws of THIS round were made at, [B, C, H, W].
            round_callback(r + 1, sigma_to_scaling(sigma, bb.T, convention))

        if average_after is not None and (r + 1) > int(average_after):
            run_sum = sigma.clone() if run_sum is None else run_sum + sigma
            run_cnt += 1
        elif average_last and average_last > 1:
            recent.append(sigma.clone())
            if len(recent) > int(average_last):
                recent.pop(0)

        del S_sum, v_sum, v_sumsq

    rounds_bar.close()

    if run_cnt > 1:
        sigma_avg = run_sum / run_cnt
        if verbose:
            shift = float((sigma_avg - sigma).abs().mean() /
                          sigma.abs().mean().clamp_min(1e-12))
            print(f"    averaged sigma over {run_cnt} post-burn-in round(s): "
                  f"mean {float(sigma.mean()):.6f} -> {float(sigma_avg.mean()):.6f} "
                  f"(MC noise removed: {shift:.3e} relative; effective samples "
                  f"~{run_cnt * N})")
        sigma = sigma_avg
    elif average_last and average_last > 1 and len(recent) > 1:
        sigma_avg = torch.stack(recent, dim=0).mean(dim=0)
        if verbose:
            shift = float((sigma_avg - sigma).abs().mean() /
                          sigma.abs().mean().clamp_min(1e-12))
            print(f"    averaged sigma over the last {len(recent)} round(s): "
                  f"mean {float(sigma.mean()):.6f} -> {float(sigma_avg.mean()):.6f} "
                  f"(MC noise removed: {shift:.3e} relative)")
        sigma = sigma_avg

    n_conv = int((~active).sum().item())
    if verbose:
        print(f"    {n_conv}/{B} sequence(s) converged (|dsigma|/sigma < {tol:g}); "
              f"rounds per sequence: min={int(rounds_run.min())} "
              f"max={int(rounds_run.max())} "
              f"mean={float(rounds_run.float().mean()):.1f}")

    # Scaling factors from the final sigma. Because the round is ordered
    # M-step -> sample, this is exactly the sigma the last draws were made at,
    # so the returned sigma, v_bar_list and u_list all describe the same
    # posterior -- and the UQ sampling in main.py continues from it.
    s_raw = sigma_to_scaling(sigma, bb.T, convention)
    scaling_factors_bb = _assemble_scaling_factors_bb(
        [s_raw[i] for i in range(B)], num_cardiac_frames, device=device,
    )

    return (scaling_factors_bb.detach(),
            sigma.detach(),
            [v.detach() for v in v_bar_list],
            [u.detach() for u in u_list],
            history,
            rounds_run.cpu())


# =========================================================================
# Total-derivative M-step (pathwise / reparameterized)
# =========================================================================

def _gauss_blur(x, sigma_px):
    """Separable Gaussian blur, depthwise, differentiable. sigma_px<=0 = identity.

    The learned ScalingFactorNetwork is a U-Net, so its sigma is spatially smooth
    by construction. A free per-voxel sigma has no such prior and picks up
    gradient noise at the voxel level, which directly corrupts the voxel RANKING
    that nAUSE scores. Blurring the pre-activation restores the smoothness
    without constraining the shape.
    """
    import torch.nn.functional as F
    if sigma_px is None or sigma_px <= 0:
        return x
    rad = max(1, int(round(3 * sigma_px)))
    t = torch.arange(-rad, rad + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma_px) ** 2)
    k = k / k.sum()
    C = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    x = F.conv2d(F.pad(x, (rad, rad, 0, 0), mode='replicate'), kx, groups=C)
    x = F.conv2d(F.pad(x, (0, 0, rad, rad), mode='replicate'), ky, groups=C)
    return x


def _expand_sf_bb(s_raw, num_cardiac_frames):
    """[B,2,H,W] -> [B,T+1,2,H,W] with a zero frame at index 0. Differentiable."""
    T = num_cardiac_frames - 1
    sf = s_raw.unsqueeze(1).expand(-1, T, -1, -1, -1)
    zero = torch.zeros_like(sf[:, 0:1])
    return torch.cat([zero, sf], dim=1), sf


def total_derivative_refine(trainer, loss_fn, source_img, target_imgs, v_reg_list,
                            rounds=200, num_samples=8, sample_chunk=0,
                            alpha=1e-4, beta=3.0, lr=1e-2, sigma_init_value=None,
                            smooth_px=0.0, log_every=25,
                            sampler='reverse_sde', langevin_kw=None,
                            langevin_convention='code_s', warm_start=False,
                            round_callback=None, smooth_weight=0.0,
                            device='cuda', progress=True, verbose=True):
    """
    Maximize Eq. 10 by the TOTAL derivative through the sampler.

    Eq. 10 depends on sigma twice: explicitly, through the transition and prior
    terms, and implicitly, through the sampled velocities v(sigma, z) that
    Eq. 14 generates with sigma inside both the drift and the injected
    sqrt(sigma)*z. The closed form `sigma_star` differentiates only the explicit
    sigma, i.e. it solves argmax_sigma p(sigma | v^r, I, v_hat) with the samples
    held fixed. Under that convention the energy term carries no sigma and drops
    out, which is exactly why the resulting sigma is blind to the image: S only
    measures frame-to-frame increments of the sampled paths, and those are
    dominated by the injected noise rather than by the data. Measured: sigma
    flat at 0.4729, spatial std/mean 0.0004, split-half correlation 0.0009 over
    ~15200 effective samples, and a hard contrast bound of 1/(1-slope) = 1.18x
    that no (alpha, beta) can widen.

    Here we take

        dL/dsigma = (dL/dsigma)|_{v fixed} + < dL/dv , dv(sigma,z)/dsigma >

    where the second term is where anatomy enters: dv/dsigma propagates the
    injected noise, and dE/dv measures how much that noise degraded alignment --
    large along edges, near zero in homogeneous regions. Averaged over z it
    behaves as ~ sigma*h(x)/gamma with h the local curvature of E, precisely the
    signal S cannot see.

    Implementation: reparameterize the samples through Eq. 14 with fixed noise z
    and back-propagate through the reverse process to sigma. This needs NO change
    to the sampler -- `run_posterior_reverse_process` already keeps sigma in the
    graph through both the drift and `sigma_z = sigma * z`, and already detaches
    grad_E (brownian_bridge.py:440-443, 314).

    Under sampler='langevin' the same construction applies to the ULA updates,
    but it is NOT on by default: the caller must set
    `langevin_kw['grad_steps']` to the number of trailing updates to track.
    Without it `langevin_velocities` detaches sigma (langevin.py:512-513) and
    runs every step under torch.no_grad(), so only the explicit
    (dL/dsigma)|_{v fixed} survives -- which is the closed form, reached by
    gradient descent instead of in one shot. With grad_steps set, sigma stays
    live through eps = step_size*sigma and sqrt(2 eps) while the score stays
    detached, mirroring the reverse SDE's detached grad_E. main.py spells this
    as --langevin_grad_steps and passes it in a SEPARATE langevin_kw from the
    one the closed-form M-step and the UQ draws use.

    sigma is a free per-sequence, per-voxel parameter here (no network): each
    image sequence gets its own [2,H,W] field, optimized by Adam on a softplus
    pre-activation so it stays positive. Swapping in the amortized
    sigma = f_theta(I, v_hat) is only a change of where s_raw comes from; the
    gradient path below is identical.

    Args:
        loss_fn: an LTMALoss built with lambda_scale=beta, alpha_scale=alpha,
                 gamma=gamma. Its compute_loss is exactly the Eq. 10 objective in
                 the 'code_s' convention: sim + gamma*(inv_gamma_prior +
                 bridge_norm + reg), i.e. gamma * [ E/gamma + S/(2 sigma) +
                 ((T-1)/2) log sigma + beta/sigma + (alpha+1) log sigma ].
        rounds:  gradient steps (not EM rounds -- there is no closed-form M-step).
        langevin_convention:
                 How the optimized variable is READ as a variance before it is
                 handed to the Langevin sampler. Ignored for sampler
                 ='reverse_sde', which takes it as `scaling_factors` s directly.

                 'paper_sigma' (the default main.py passes) hands over 2s/T,
                 matching what the reverse SDE actually realizes internally
                 (brownian_bridge.py:414). Measured kappa = 0.149 against the
                 reverse SDE's 0.182, dispersion 0.88x, loss 0.00398 against
                 0.00404 -- i.e. the two samplers become comparable and a
                 Langevin run can stand in for a reverse-SDE one.

                 'code_s' passes s unchanged. That is self-consistent with the
                 loss -- losses.py's bridge_norm assumes increments of variance
                 s*rho_t, and the chain delivers exactly that (measured
                 kappa = 0.87 at 100 steps, -> 1 at stationarity). But the
                 reverse SDE does NOT, so at equal s the chain disperses 1.67x
                 more, its trajectories sit further from alignment, and the
                 similarity term reads correspondingly higher (0.00476). Read
                 that loss only against other 'code_s' runs.

                 This is the signature default='code_s' rather than
                 'paper_sigma' so that calling the function directly preserves
                 the raw pass-through; main.py opts into the matched one. Same
                 trap compare_samplers_figure.py's header documents.
        warm_start:
                 sampler='langevin' only. True carries each chain's final state
                 over to the next gradient step (persistent chains), so step
                 r+1 resumes where step r stopped instead of re-initializing
                 per langevin_kw['init']. Only step 1 uses `init`. The carried
                 state is detached, so the pathwise gradient still runs through
                 the current step's tracked updates alone. One state is kept per
                 sample chunk, i.e. N*B chains in total.
        smooth_weight:
                 Weight on an l2 SPATIAL SMOOTHNESS penalty on sigma, ported
                 from uncertainty_sde_acdc_combined_iclr_2/main_langevin.py's
                 --sigma_smooth_weight. A free per-voxel sigma is H*W
                 independent scalars with nothing in the loss coupling
                 neighbours, so it comes out speckled; the learned U-Net is
                 smooth by construction. losses.Grad(penalty='l2') on sigma
                 itself (not s_pre), added AFTER compute_loss, so the logged
                 similarity / bridge_norm / inv_gamma_prior stay exactly what
                 LTMALoss reports; the penalty is logged as 'sigma_smooth'.
                 0 = off.

    Returns (scaling_factors_bb, sigma, v_bar_list, u_list, history, steps).
    """
    import torch.nn.functional as F

    num_cardiac_frames = len(v_reg_list)
    T = num_cardiac_frames - 1
    N = max(1, int(num_samples))
    B = v_reg_list[0].shape[0]
    chunk = int(sample_chunk) if int(sample_chunk) > 0 else N

    s0 = float(sigma_init_value if sigma_init_value is not None
               else sigma_init(alpha, beta, T))
    # softplus^{-1}(s0), so the optimization starts exactly at the closed-form
    # sigma^1 and any structure that appears is attributable to the new term.
    raw0 = float(np.log(np.expm1(s0))) if s0 > 0 else -1.0
    s_pre = torch.full_like(v_reg_list[0].detach().float(), raw0).requires_grad_(True)
    opt = torch.optim.Adam([s_pre], lr=lr)

    history = []
    v_bar_list = [v.detach().float().clone() for v in v_reg_list]
    u_list = [torch.zeros_like(v) for v in v_bar_list]

    # Persistent Langevin chains: final latent eta of each sample chunk, keyed by
    # the chunk's offset `drawn`, fed back as init_eta on the next step.
    warm = bool(warm_start) and sampler == 'langevin'
    chain_state = {}

    if verbose:
        print(f"    total-derivative M-step: B={B}, T={T}, N={N} (chunk {chunk}), "
              f"steps={rounds}, lr={lr:g}"
              + ("  [langevin warm start]" if warm else ""))
        print(f"    sigma^1 = {s0:.6f} (constant), alpha={alpha:g}, beta={beta:g}")

    bar = tqdm(range(int(rounds)), desc="  total-deriv steps",
               leave=False, position=1, disable=not progress)

    for r in bar:
        opt.zero_grad(set_to_none=True)
        sigma = F.softplus(_gauss_blur(s_pre, smooth_px))

        sums = [None] * num_cardiac_frames
        sumsq = [None] * num_cardiac_frames
        total_val = 0.0
        parts = {}
        drawn = 0
        while drawn < N:
            n_c = min(chunk, N - drawn)
            w = n_c / N

            def _rep(t):
                return t.repeat(n_c, *([1] * (t.dim() - 1)))

            sf_bb, sf = _expand_sf_bb(sigma, num_cardiac_frames)
            v_reg_C = [_rep(v) for v in v_reg_list]
            src_C = _rep(source_img)
            tgt_C = [_rep(t) for t in target_imgs]

            # The optimized variable IS the reverse SDE's `scaling_factors` s
            # (that is what _expand_sf_bb feeds it). The Langevin sampler is
            # parameterized directly in sigma instead, so handing it s unchanged
            # ('code_s') gives it the T/2 = 7x wider prior -- see the
            # langevin_convention note in the docstring. A plain division, so
            # the pathwise gradient to s_pre is unaffected.
            sigma_lg = scaling_to_sigma(sigma, trainer.brownian_bridge.T,
                                        langevin_convention)

            # sigma flows into the SDE here -- no detach anywhere on this path.
            lkw = langevin_kw
            if warm:
                lkw = dict(langevin_kw or {}, return_latent=True)
                if drawn in chain_state:
                    lkw['init_eta'] = chain_state[drawn]
            out = draw_velocities(
                sampler, trainer, v_reg_C, _rep(sf_bb), src_C, tgt_C,
                num_samples=n_c, sigma=_rep(sigma_lg),
                langevin_kw=lkw,
            )
            if warm:
                # xi is already detached (langevin.py), so no graph crosses steps.
                v_t_list_C, _, chain_state[drawn] = out
            else:
                v_t_list_C, _ = out
            warped_C = [trainer.warp_image(src_C, v) for v in v_t_list_C]

            loss_C, ld_C = loss_fn.compute_loss(
                src_C, tgt_C, warped_C[1:], v_t_list_C[1:], _rep(sf),
                v_t_list=v_t_list_C, v_reg_list=v_reg_C,
                num_cardiac_frames=num_cardiac_frames,
            )
            (loss_C * w).backward()

            with torch.no_grad():
                for fi, v_C in enumerate(v_t_list_C):
                    vv = v_C.detach().float().view(n_c, B, *v_C.shape[1:])
                    c = vv - v_reg_list[fi].detach().float().unsqueeze(0)
                    s1, s2 = c.sum(dim=0), (c * c).sum(dim=0)
                    sums[fi] = s1 if sums[fi] is None else sums[fi] + s1
                    sumsq[fi] = s2 if sumsq[fi] is None else sumsq[fi] + s2

            total_val += float(loss_C.detach()) * w
            for k, v in ld_C.items():
                parts[k] = parts.get(k, 0.0) + float(v) * w
            del v_t_list_C, warped_C, loss_C, v_reg_C, src_C, tgt_C, sf_bb, sf
            drawn += n_c

        # Smoothness penalty, once per step and on a fresh sigma graph (the
        # chunk backward()s above have already freed theirs).
        if smooth_weight > 0:
            pen = smooth_weight * _SIGMA_GRAD.loss(
                F.softplus(_gauss_blur(s_pre, smooth_px)))
            pen.backward()
            parts['sigma_smooth'] = float(pen.detach())
            total_val += parts['sigma_smooth']

        opt.step()

        with torch.no_grad():
            sigma = F.softplus(_gauss_blur(s_pre, smooth_px))
            for fi in range(num_cardiac_frames):
                mean_c = sums[fi] / N
                v_bar_list[fi] = v_reg_list[fi].detach().float() + mean_c
                u_list[fi] = (sumsq[fi] / N - mean_c ** 2).clamp_min(0.0).sqrt()
            rec = {
                'step': r + 1,
                'loss': total_val,
                'similarity': parts.get('similarity', 0.0),
                'bridge_norm': parts.get('bridge_norm', 0.0),
                'inv_gamma_prior': parts.get('inv_gamma_prior', 0.0),
                'sigma_smooth': parts.get('sigma_smooth', 0.0),
                'sigma_mean': float(sigma.mean()),
                'sigma_std': float(sigma.std()),
                'sigma_min': float(sigma.min()),
                'sigma_max': float(sigma.max()),
            }
        history.append(rec)
        bar.set_postfix({'loss': f"{total_val:.4f}",
                         'sig': f"{rec['sigma_mean']:.4f}",
                         'std': f"{rec['sigma_std']:.4f}",
                         **({'smooth': f"{rec['sigma_smooth']:.2e}"}
                            if smooth_weight > 0 else {})})
        if verbose and (log_every <= 1 or (r + 1) % log_every == 0 or r == 0):
            print(f"    step {r + 1:>4}: loss={total_val:.5f} "
                  f"sim={rec['similarity']:.5f}  sigma mean={rec['sigma_mean']:.5f} "
                  f"std={rec['sigma_std']:.5f} "
                  f"[{rec['sigma_min']:.5f}, {rec['sigma_max']:.5f}]")
        if round_callback is not None:
            # s after this step's Adam update, [B, C, H, W].
            round_callback(r + 1, torch.clamp(sigma, S_MIN, S_MAX))
        del sums, sumsq

    bar.close()

    with torch.no_grad():
        sigma = F.softplus(_gauss_blur(s_pre, smooth_px)).detach()
        sf_bb, _ = _expand_sf_bb(torch.clamp(sigma, S_MIN, S_MAX), num_cardiac_frames)

    return (sf_bb.detach(), sigma, [v.detach() for v in v_bar_list],
            [u.detach() for u in u_list], history,
            torch.full((B,), int(rounds), dtype=torch.long))
