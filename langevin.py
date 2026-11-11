"""
Langevin sampler for the velocity posterior, as an alternative to the Eq. 14
reverse SDE.

FULL COPY of `uncertainty_sde_combined_acdc_v3_iclr/langevin.py` (identical
code; this paragraph is the only difference), so the sampler the mcem package
runs -- including the preconditioning in _diag_precond, _apply_K and _apply_A --
can be read here. It still imports `warping` and `losses` from
`uncertainty_sde_combined_acdc_v3_iclr`, so no import cycle is created (that
package's trainer keeps using its own copy). The two files, and
`uncertainty_sde_acdc_combined_iclr_2/langevin.py`, must be kept in sync by hand.

Target
------
The same posterior the reverse SDE approximates,

    log p(v | sigma, I, v_hat) = -E(v)/gamma - S(eta)/(2 sigma) + const,
    eta_t = v_t - v_hat_t,   eta_0 = eta_T = 0,

where the second term is the Brownian bridge prior written out through its
Prop. 1(iii) transitions eta_t | eta_{t-1} ~ N(rho_t eta_{t-1}, sigma rho_t),
so that

    -log p_prior(eta) = sum_{t=1}^{T-1} (eta_t - rho_t eta_{t-1})^2 / (2 sigma rho_t)
                      = S(eta) / (2 sigma),        rho_t = (T-t)/(T-(t-1)).

That S is exactly `mcem.bridge_residual_S`. The t = T transition is degenerate
(rho_T = 0 pins eta_T = 0 regardless of eta_{T-1}) so it contributes no term,
which is the same exclusion the closed-form M-step makes.

Sampling
--------
Unadjusted Langevin (ULA) on the free coordinates eta_1..eta_{T-1}:

    eta <- eta + (eps/2) * [ -grad_eta E(v_hat+eta)/gamma - grad_eta S(eta)/(2 sigma) ]
               + sqrt(eps) * z

with eta_0 and eta_T held at 0 every step. eps is preconditioned as
eps = step_size * sigma, which makes the prior term well conditioned: for a
Gaussian of variance ~sigma*rho, ULA is stable while eps < 2 sigma rho, i.e.
while step_size < 2 rho, and rho ranges over [1/2, (T-1)/T]. step_size <= 0.5 is
therefore safe across all frames; the default is deliberately below that.

`scale_step_with_sigma=False` turns that preconditioning off and runs the plain
discretization eps = step_size, with the noise sqrt(2 step_size) to match. The
step is then an ABSOLUTE size shared by every voxel and frame rather than a
dimensionless stability margin, so the stability condition eps < 2 sigma rho has
to be met by the raw value -- a step_size tuned for the scaled mode is orders of
magnitude too large here. Off by default; see main_langevin's
--no_langevin_scale_step.

Both gradients come from autograd rather than being hand-derived -- S is
differentiable as written, and E needs the VecInt + grid_sample adjoint anyway.

Two independent smoothing knobs
-------------------------------
`smooth_width` smooths the FIELD: the sampled variable becomes a latent white
bridge xi and the velocity is v = v_hat + K*xi. The prior and the injected noise
stay white (on xi), the target density is unchanged in form, and the closed-form
M-step still applies verbatim.

`noise_smooth_width` instead makes K the PRECONDITIONER of the step, through the
matrix A = sigma^(1/2) K sigma^(1/2):

    eta <- eta + eps * A grad log p + sqrt(2 eps) * A^(1/2) z
    A = S K S,  S = diag(sigma^(1/2))

A is symmetric positive definite and factors as (S K^(1/2))(K^(1/2) S), so
B = S K^(1/2) is a square root: the noise is drawn as sqrt(2 eps) * S K^(1/2) z,
which has Cov = 2 eps * S K S = 2 eps A without ever forming A^(1/2). K^(1/2) is
the Gaussian of width w/sqrt(2), since G_a * G_a = G_{a sqrt 2}. In code:

    def apply_A(g, sigma, gaussian_blur):
        s = sigma.sqrt()                    # elementwise, same shape as g
        return s * gaussian_blur(s * g)

The SANDWICH is what makes this work: the same operator A drives the drift and
the noise covariance, and a preconditioned chain leaves p invariant only when it
does. Two earlier versions of this knob got that wrong -- the first blurred ONLY
z and left the drift at eps * grad log p; the second fixed the noise but used
S^2 K for the drift, which is not symmetric once sigma varies per voxel (and it
always does: sigma is a per-voxel [B,C,H,W] map). Both were defects, not
trade-offs, and A = S K S is the fix.

WHAT IT DOES AND DOES NOT DO. Preconditioning changes the GEOMETRY of the walk,
not the target -- dv = P grad log p dt + sqrt(2P) dW has stationary density p for
any constant P. So K does not make the sampled fields smoother in the
exact-sampling limit, and the section-2 M-step consistency is preserved rather
than broken. What it changes is mixing (the warp low-passes velocity, so
correlated moves are the ones the likelihood can feel) and, at short chains like
the default 14 steps, the finite-step transient -- which is nowhere near
stationarity and is visibly shaped by K.

THE SEED MATTERS, and is handled. A Gaussian K is numerically singular -- it
annihilates the highest frequencies (6 of 64 eigenvalues below 1e-12 at w=4).
Those modes get zero drift AND zero noise, so the chain never moves them and
whatever the initial draw put there is frozen for the whole run: measured,
corr(eta_final, eta_init) = 0.988 after 14 steps and 0.939 after 200, against
0.301 for the plain sampler. With the white `_prior_draw` that made every
roughness metric report the SEED rather than the sampler (njn% 33% at w=4,
tracing the draw's own bridge variance arch). The prior draw is therefore
smoothed by K^(1/2) at init -- see the `init == 'prior'` branch below.

One numerical caveat left, measured on a 64-point axis: `smooth_field` pads with
'replicate', which is NOT a symmetric operator (|K - K^T| = 2.2e-1 at w=2).
Strictly K should be symmetric for the scheme to be exact; zero padding (0.0) or
a circular/FFT blur (4e-17) would be. Separately, if the chain should explore the
rough directions at all rather than freezing them at zero, that needs a ridge
K <- (K + lam I)/(1 + lam), lam ~ 0.01 (measured stationary-covariance error
0.52 -> 2e-4 at w=4) -- but the posterior it would then explore is rough, which
is the opposite of what this knob is for.

If smoothness must be a property of the POSTERIOR rather than of the walk, that
is `smooth_width` above (v = v_hat + K*xi), which changes the model. Both knobs
are independent and may be combined; both are off (0.0) by default.

Why this is not just a swap
---------------------------
Two differences from `run_posterior_reverse_process` that matter:

1. NO Eq. 13 APPROXIMATION. The reverse SDE needs a likelihood score at each
   intermediate diffusion step, where every other frame is marginalized; that is
   intractable, so Chung et al.'s approximation evaluates the energy at the
   bridge conditional means mu_{s|t} of all frames. Langevin moves the whole
   trajectory jointly, so grad E is the exact gradient of the actual energy --
   no surrogate. There is no opt-in to the surrogate: that comparison mode was
   removed, and `likelihood` now accepts only 'exact'.

2. CONSISTENT WITH THE M-STEP. The stationary density here is by construction
   the one whose prior is S(eta)/(2 sigma), i.e. precisely the model the
   closed-form M-step assumes. So E[S] = (T-1) sigma, kappa = 1, and the EM map
   has slope (T-1)/(2 alpha + T + 1) = 0.867 with fixed point beta/(alpha+1) --
   the behaviour predicted analytically, which the Euler-Maruyama sampler does
   not deliver (measured kappa there: 1.2454 in paper units, 0.178 in code
   units; see mcem.py). This is a property of the sampler, not a tuning choice.

Caveats. ULA is asymptotically biased for finite eps (no Metropolis correction),
and `n_steps` chains started from a prior draw are only approximately converged.
Both are the usual ULA trade-offs and are exposed as knobs rather than hidden.
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import math

import torch
import torch.nn.functional as F

from uncertainty_sde_combined_acdc_v3_iclr.warping import warp_image_with_velocity
from uncertainty_sde_combined_acdc_v3_iclr.losses import Grad

_GRAD = Grad(penalty='l2')


class _NoGrad(torch.no_grad):
    """Alias, so the output block can pick its context by name."""


class _KeepGrad:
    """No-op context: keeps the autograd graph when grad_steps > 0."""
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def bridge_ratio(t, T):
    """rho_t = (T - t) / (T - (t - 1))."""
    return (T - t) / (T - (t - 1))


def _diag_precond(sigma, source_img, gamma, guidance_scale, reg_weight=0.0):
    """
    Constant diagonal preconditioner M(x) = 1 / (1/sigma + lik_precision(x)).

    The stiffness is diagonal and known in closed form: the bridge prior
    contributes precision ~1/(sigma rho) (roughly uniform), while the likelihood
    contributes 2 g ||grad I||^2 / (gamma H W), which is enormous at edges and
    ~0 in flat regions. A single scalar step therefore cannot serve both -- it is
    either unstable at edges or glacial elsewhere, which is what the gamma sweep
    ran into (usable only down to 5e-6, NaN by 5e-8).

    Stepping with eps(x) = step * M(x) equalizes the two. Because M depends only
    on the image and sigma -- never on eta -- it is a CONSTANT preconditioner, so
    preconditioned Langevin still targets p exactly:

        eta <- eta + (eps/2) grad log p + sqrt(eps) z,     eps = step * M.

    Returned shape broadcasts over the frame axis: [B, 1, C, H, W].
    """
    I = source_img
    gy = torch.zeros_like(I); gx = torch.zeros_like(I)
    gy[:, :, 1:-1, :] = 0.5 * (I[:, :, 2:, :] - I[:, :, :-2, :])
    gx[:, :, :, 1:-1] = 0.5 * (I[:, :, :, 2:] - I[:, :, :, :-2])
    g2 = (gx ** 2 + gy ** 2)                       # [B,1,H,W]
    n_pix = I.shape[-1] * I.shape[-2]
    lik = 2.0 * guidance_scale * g2 / (gamma * n_pix)
    lik = lik.expand(-1, sigma.shape[1], -1, -1)   # match channels
    M = 1.0 / (1.0 / sigma.clamp_min(1e-8) + lik)
    return M.unsqueeze(1)


def _gauss_kernel_1d(width, device, dtype):
    """Normalized 1-D Gaussian of std `width`, radius 3*width. Returns (k, r)."""
    r = max(1, int(round(3 * width)))
    t = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t / width) ** 2)
    return k / k.sum(), r


def smooth_field(x, width):
    """Gaussian blur K applied per channel. width<=0 is the identity."""
    if width is None or width <= 0:
        return x
    k, r = _gauss_kernel_1d(width, x.device, x.dtype)
    C = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='replicate'), kx, groups=C)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode='replicate'), ky, groups=C)
    return x


def _half_kernel_ops(width, H, W, device, dtype):
    """
    Zero-padded half-blur Kh and the per-pixel gain N that restores the interior
    noise variance at the border. Returns (k, r, N) with N of shape [H, W].

    Kh is a zero-padded convolution with a symmetric kernel, i.e. a symmetric
    Toeplitz operator: exact adjoint, and NO coupling between opposite edges.
    Near the border it drops the out-of-image taps, so the row norm
    ||Kh[x, :]||^2 = a_i * a_j (separable) falls below the interior value
    a^2, a = sum k^2. N(x) = sqrt(a^2 / (a_i a_j)) rescales each pixel back to
    the interior variance.
    """
    k, r = _gauss_kernel_1d(width, device, dtype)
    k2 = k * k
    a = k2.sum()

    def row_norm(n):
        # a_i = sum of k^2 over the taps that land inside [0, n)
        ones = torch.ones(1, 1, n, device=device, dtype=dtype)
        return F.conv1d(ones, k2.view(1, 1, -1), padding=r).view(n)

    ai, aj = row_norm(H), row_norm(W)
    N = torch.sqrt(a * a / (ai.view(H, 1) * aj.view(1, W)))
    return k, r, N


def _blur_zero(x, k, r):
    """Separable zero-padded blur of [N, C, H, W] -- symmetric, no wraparound."""
    C = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    x = F.conv2d(x, kx, padding=(0, r), groups=C)
    x = F.conv2d(x, ky, padding=(r, 0), groups=C)
    return x


def _apply_K(x, width, transpose=False):
    """
    The preconditioner's square-root factor Kn = N Kh over the spatial axes of a
    [B, T+1, C, H, W] chain state (transpose=True applies Kn^T = Kh N).

    Boundary handling. The first version reused smooth_field, whose 'replicate'
    padding is not symmetric (<Kx,y> != <x,Ky>, 1.97 vs -2.75 at width 2) and
    gave the noise 2.5x the interior variance on edges and 5.6x in corners, so
    the chain did not target the posterior near the border -- a bright frame
    in sigma. Circular padding fixed the symmetry but WRAPS: a strong effect on
    one edge leaked onto the opposite one (measured sigma 4.57 on a top-edge
    pixel with a matching spot on the bottom edge). Here Kh is zero-padded
    (symmetric, no wraparound) and N restores the interior variance per pixel.
    The noise is Kn z with Cov = Kn Kn^T, and _apply_A uses exactly
    Kn Kn^T for the drift, so Cov(noise) = 2 eps * (drift operator) holds
    everywhere by construction.

    No global variance renormalization -- deliberately. An earlier version
    divided the blurred noise by its own std to reach unit per-voxel variance;
    the noise must realize the drift operator's covariance, which it does.

    width <= 0 is the identity.
    """
    if not width or width <= 0:
        return x
    B_, Tp1 = x.shape[0], x.shape[1]
    H, W = x.shape[-2], x.shape[-1]
    k, r, N = _half_kernel_ops(width, H, W, x.device, x.dtype)
    y = x.reshape(B_ * Tp1, *x.shape[2:])
    if transpose:
        return _blur_zero(y * N, k, r).reshape(x.shape)
    return (N * _blur_zero(y, k, r)).reshape(x.shape)


def _apply_A(g, sigma, width):
    """
    A g = sigma^(1/2) K sigma^(1/2) g -- the preconditioning matrix of the step

        v <- v + eps A grad log p + sqrt(2 eps) A^(1/2) z.

    S = diag(sigma^(1/2)) sits on BOTH sides of the blur, and that is the whole
    point. A = S K S is symmetric positive definite, and it factors as
    (S K^(1/2))(K^(1/2) S), so B = S K^(1/2) is a square root and B z has
    Cov = S K S = A exactly -- which is how the noise term realizes A^(1/2)
    without ever forming it. The earlier form S^2 K is NOT symmetric once sigma
    varies per voxel (and it always does: sigma is a per-voxel [B,C,H,W] map),
    so drift and noise then disagreed and the chain targeted nothing.

    Called with sigma = eps (= step_size * M), so the returned field is the
    whole eps-scaled drift: sqrt(eps) K sqrt(eps) g = step_size * S K S g.

    width <= 0 is K = I, where S K S = diag(sigma) and this collapses to the
    plain sigma * g -- the unsmoothed path, byte-for-byte unchanged.
    """
    if not width or width <= 0:
        return sigma * g
    s = torch.sqrt(torch.clamp(sigma, min=0.0))   # elementwise, broadcasts over T
    # A = S Kn Kn^T S with Kn = N Kh the SAME factor the noise uses (half width
    # w/sqrt2), so the drift operator is B B^T with B = S Kn exactly and
    # Cov(noise) = 2 eps * (drift operator) holds by construction, border
    # included. See _apply_K for the boundary handling.
    h = width / math.sqrt(2.0)
    return s * _apply_K(_apply_K(s * g, h, transpose=True), h)


def _neg_log_prior(eta, sigma, T):
    """
    S(eta) / (2 sigma), summed to a scalar.

    eta:   [B, T+1, C, H, W]
    sigma: [B, C, H, W], per voxel and per channel

    Per-voxel terms are divided by that voxel's sigma BEFORE summing, so the
    spatially varying sigma enters correctly rather than as an average.
    """
    tot = eta.new_zeros(())
    for t in range(1, T):
        rho = bridge_ratio(t, T)
        r = eta[:, t] - rho * eta[:, t - 1]
        tot = tot + (r * r / (2.0 * sigma * rho)).sum()
    return tot


def _neg_log_prior_per_chain(eta, sigma, T):
    """S(eta)/(2 sigma) reduced to one scalar per chain -> [B]."""
    tot = None
    for t in range(1, T):
        rho = bridge_ratio(t, T)
        r = eta[:, t] - rho * eta[:, t - 1]
        term = (r * r / (2.0 * sigma * rho)).flatten(1).sum(1)
        tot = term if tot is None else tot + term
    return tot


def _energy_per_chain(v_list, source_img, target_imgs, inshape, reg_weight,
                      frame_chunk=0):
    """
    E(v) as one scalar per chain -> [B]. Same terms as _energy.

    The frames are STACKED along the batch axis and warped in one call rather
    than looped over. At B=32, T=14, 64x64 the loop costs 121 ms per evaluation
    against 13.9 ms stacked -- 8.7x -- because each warp is a VecInt of seven
    grid_samples on a tiny tensor, so the step is bound by kernel launches, not
    arithmetic. That energy term is ~60% of a chain step's forward and the
    chain is >99% of training time, so this is the single largest win available.

    frame_chunk bounds how many frames are stacked at once (0 = all). Peak
    memory scales with chunk*B rows; the sum over frames is exact either way.
    """
    Tn = len(target_imgs)
    B = source_img.shape[0]
    chunk = frame_chunk if frame_chunk and frame_chunk > 0 else Tn
    rep = (1,) * (source_img.dim() - 1)

    E = None
    for start in range(0, Tn, chunk):
        js = range(start, min(start + chunk, Tn))
        n = len(js)
        # frame-major: row index = local_frame * B + b
        v_all = torch.cat([v_list[j + 1] for j in js], dim=0)
        tgt_all = torch.cat([target_imgs[j] for j in js], dim=0)
        src_all = source_img.repeat(n, *rep)

        warped = warp_image_with_velocity(src_all, v_all, inshape=inshape)
        e = ((warped - tgt_all) ** 2).mean(dim=(1, 2, 3))          # [n*B]
        if reg_weight:
            d = (v_all[:, :, 1:, :] - v_all[:, :, :-1, :]) ** 2
            dx = (v_all[:, :, :, 1:] - v_all[:, :, :, :-1]) ** 2
            e = e + reg_weight * 0.5 * (d.mean(dim=(1, 2, 3)) + dx.mean(dim=(1, 2, 3)))
        e = e.view(n, B).sum(dim=0)                                # [B]
        E = e if E is None else E + e
    return E


def _logp_per_chain(eta, v_hat, sigma, source_img, target_imgs, inshape,
                    gamma, guidance_scale, reg_weight, T, smooth_width=0.0):
    """log p(xi) up to a constant, per chain -> [B].

    With smooth_width > 0 the sampled variable `eta` is the LATENT white bridge
    xi, and the velocity is v = v_hat + K*xi. The prior term is unchanged (it is
    the standard diagonal bridge on xi, so the closed-form M-step still applies
    verbatim); only the likelihood sees the smoothed field. This is the LDDMM
    construction, and it exists because the warp low-passes velocity: at matched
    magnitude a width-2 smoothed perturbation changes E by 0.607 vs 0.127 for
    white noise (4.8x), so the likelihood can actually resist it.
    """
    B_, Tp1 = eta.shape[0], eta.shape[1]
    if smooth_width and smooth_width > 0:
        e = smooth_field(eta.reshape(B_ * Tp1, *eta.shape[2:]), smooth_width)
        e = e.reshape(B_, Tp1, *eta.shape[2:])
    else:
        e = eta
    v = v_hat + e
    v_list = [v[:, t] for t in range(T + 1)]
    E = _energy_per_chain(v_list, source_img, target_imgs, inshape, reg_weight)
    P = _neg_log_prior_per_chain(eta, sigma, T)
    return -(guidance_scale / gamma) * E - P


def _energy(v_list, source_img, target_imgs, inshape, reg_weight):
    """
    E(v) = sum_t [ mean_pixels (warp(src, v_t) - I_t)^2 + reg_weight * Grad(v_t) ]

    Reduced as mean over pixels but SUM over the batch, so each sequence's
    gradient is its own and does not depend on how many sequences share the
    batch (the 1/B_eff issue documented in mcem.PerSequenceBridge).
    """
    E = v_list[0].new_zeros(())
    for j, tgt in enumerate(target_imgs):
        v_t = v_list[j + 1]
        warped = warp_image_with_velocity(source_img, v_t, inshape=inshape)
        E = E + ((warped - tgt) ** 2).mean(dim=(1, 2, 3)).sum()
        if reg_weight:
            d = (v_t[:, :, 1:, :] - v_t[:, :, :-1, :]) ** 2
            dx = (v_t[:, :, :, 1:] - v_t[:, :, :, :-1]) ** 2
            E = E + reg_weight * 0.5 * (d.mean(dim=(1, 2, 3)).sum()
                                        + dx.mean(dim=(1, 2, 3)).sum())
    return E


def _prior_draw(v_reg_list, sigma, T, generator=None):
    """
    Exact draw from the bridge prior via the Prop. 1(iii) forward recursion
    eta_t = rho_t eta_{t-1} + sqrt(sigma rho_t) z, used as the ULA init.
    eta_0 = 0 by construction and eta_T = 0 because rho_T = 0.
    """
    B = v_reg_list[0].shape[0]
    eta = [torch.zeros_like(v_reg_list[0])]
    for t in range(1, T + 1):
        rho = bridge_ratio(t, T)
        if rho <= 0:
            eta.append(torch.zeros_like(eta[-1]))
            continue
        z = torch.randn(eta[-1].shape, device=eta[-1].device, dtype=eta[-1].dtype,
                        generator=generator)
        eta.append(rho * eta[-1] + torch.sqrt(sigma * rho) * z)
    return torch.stack(eta, dim=1)          # [B, T+1, C, H, W]


# Running MALA accept counts over every langevin_velocities call, so callers that
# go through draw_velocities (which drops return_stats) can still report the
# acceptance rate. Read and reset with pop_mala_stats().
MALA_STATS = {'acc': 0, 'prop': 0}


def pop_mala_stats():
    """Acceptance rate since the last call (nan if nothing was proposed) and the
    number of proposals it covers; resets the counters."""
    acc, prop = MALA_STATS['acc'], MALA_STATS['prop']
    MALA_STATS['acc'] = MALA_STATS['prop'] = 0
    return (acc / prop if prop else float('nan')), prop


def langevin_velocities(v_reg_list, sigma, source_img, target_imgs,
                        gamma=5e-5, n_steps=100, step_size=0.05,
                        reg_weight=0.1, inshape=None, init='prior',
                        guidance_scale=1.0, generator=None, verbose=False,
                        precondition='none', mala=False, return_stats=False,
                        smooth_width=0.0, noise_smooth_width=0.0,
                        renorm_K=False,
                        grad_bypass_K=False,
                        return_latent=False,
                        pin_v_endpoints=False, likelihood='exact',
                        eq13_chunk=0, init_eta=None, grad_steps=0,
                        collect_every=0, scale_step_with_sigma=True):
    """
    ULA on p(v | sigma, I, v_hat). Drop-in for
    `trainer.compute_brownian_bridge_velocities`.

    Args:
        v_reg_list: list of T+1 tensors [B, C, H, W] -- v_hat, index 0 pinned.
        sigma:      [B, C, H, W] in the SAME units the M-step solves for, i.e.
                    delta_{t|t-1} = sigma * rho_t. NOT the reverse SDE's
                    `scaling_factors` unless the convention is 'code_s'.
        n_steps:    ULA iterations. Each costs one energy gradient.
        step_size:  eps = step_size * sigma. Keep <= 0.5 (see module docstring).
                    With scale_step_with_sigma=False it is instead the absolute
                    step eps itself, and must be chosen on the scale of sigma.
        scale_step_with_sigma:
                    True (default) multiplies the step by the preconditioner M
                    (sigma, or the diagonal preconditioner), as in every run so
                    far. False drops sigma from BOTH terms of the update:
                    eps = step_size, noise sqrt(2 step_size). Note this also
                    removes the sampler's pathwise dL/dsigma -- see the comment
                    at the tracked step below.
        noise_smooth_width:
                    Width of the Gaussian kernel K used to PRECONDITION the step.
                    0 (default) is the plain isotropic update, byte-for-byte the
                    previous behaviour. With w > 0 the preconditioner is the
                    matrix A = sigma^(1/2) K sigma^(1/2) = S K S:

                        eta <- eta + eps * A grad log p
                                   + sqrt(2 eps) * A^(1/2) z

                    The drift is sqrt(eps) K sqrt(eps) grad log p -- S on BOTH
                    sides of the blur -- and the noise is sqrt(2 eps) S K^(1/2) z,
                    where K^(1/2) is the Gaussian of width w/sqrt(2) because
                    G_a * G_a = G_{a sqrt 2}. Since A = (S K^(1/2))(K^(1/2) S),
                    that noise has Cov = 2 eps A exactly, matching the drift
                    operator -- the property the noise-only and the S^2 K
                    versions both broke.

                    This changes the GEOMETRY of the walk, not the target. The
                    stationary density is still p(v | sigma, I, v_hat), the
                    section-2 M-step consistency still holds, and the sampled
                    fields are NOT smoother in the exact-sampling limit. What it
                    buys is mixing along the directions the warp can feel, plus a
                    differently shaped finite-step transient -- which at the
                    default 14-step chain is most of what you observe. For
                    smoothness in the target itself, use `smooth_width`.

                    Independent of `smooth_width`; both may be set at once.
                    Incompatible with mala=True.
        likelihood: 'exact' only. The Eq. 13 surrogate has been removed; the
                    score is always the exact gradient of the joint log
                    posterior at v_hat + eta. Kept as a parameter so callers
                    that still pass it (mcem/main.py's --langevin_likelihood)
                    fail loudly instead of silently getting different math.
        eq13_chunk: accepted and ignored, for the same reason.
        init_eta:   resume from an existing chain state ([B,T+1,C,H,W], or the
                    list of T+1 tensors that `return_latent` hands back) instead
                    of re-initializing. After burn-in the chain is already at
                    stationarity, so additional samples come from running it on
                    with a short thinning interval -- restarting from the prior
                    each time would pay the burn-in cost again for every draw.
                    Overrides `init`.

    Returns:
        (v_t_list, std_tilde_list) matching compute_brownian_bridge_velocities:
        v_t_list is a list of T+1 tensors [B, C, H, W].
    """
    if likelihood != 'exact':
        raise ValueError(
            f"likelihood={likelihood!r} is no longer supported: the Eq. 13 "
            f"surrogate path was removed from this sampler. Only 'exact' "
            f"remains -- the score is the gradient of the true joint log "
            f"posterior. Drop --langevin_likelihood (or set it to 'exact').")
    if noise_smooth_width and noise_smooth_width > 0 and mala:
        raise ValueError(
            f"noise_smooth_width={noise_smooth_width:g} is incompatible with "
            f"mala=True: under the preconditioner P = S K S the proposal is "
            f"N(mean, 2 step P), so the accept ratio needs P^-1 in lq_fwd / "
            f"lq_rev, where this code uses the isotropic ||.||^2 / (4 eps). K "
            f"is numerically singular at these widths (a Gaussian blur kills "
            f"the high frequencies outright), so P^-1 is not something to form "
            f"here. Use ULA (drop --langevin_mala) -- it is already exact in "
            f"the limit for this preconditioner -- or drop "
            f"--langevin_noise_smooth_width.")

    T = len(v_reg_list) - 1
    if inshape is None:
        inshape = (v_reg_list[0].shape[-2], v_reg_list[0].shape[-1])

    v_hat = torch.stack([v.detach() for v in v_reg_list], dim=1)  # [B,T+1,C,H,W]
    # grad_steps > 0 keeps sigma ATTACHED so the last grad_steps updates carry a
    # pathwise gradient back to it -- see the block below the loop.
    if not grad_steps:
        sigma = sigma.detach()

    if init_eta is not None:
        # Continue an existing chain. Once burn-in is done the chain is at
        # stationarity, so further samples are drawn by running it on rather
        # than restarting from the prior -- `init` is ignored in this case.
        eta = (torch.stack(init_eta, dim=1) if isinstance(init_eta, (list, tuple))
               else init_eta)
    elif init == 'prior':
        # WHITE bridge draw, used as-is -- no K applied to the seed.
        #
        # Note what this means under K-preconditioning: every increment the
        # chain makes passes through K, so it cannot remove roughness the seed
        # already has. The white high-frequency content is therefore frozen for
        # the whole run -- measured corr(eta_final, eta_init) = 0.988 after 14
        # steps and still 0.939 after 200, against 0.301 for the plain sampler.
        # Roughness metrics on eta then largely report this draw rather than the
        # sampler (njn% ~33% at w=4, tracing the draw's own bridge arch).
        #
        # Kept white deliberately: it is the exact bridge prior the closed-form
        # M-step is defined against, and smoothing it changed the seed's scale
        # (blur-only collapsed kappa = S/((T-1) sigma) from 5.04 to 0.068).
        eta = _prior_draw(v_reg_list, sigma, T, generator=generator)
    else:
        eta = torch.zeros_like(v_hat)
    eta = eta.detach()
    if pin_v_endpoints:
        eta[:, 0] = -v_hat[:, 0]
        eta[:, T] = -v_hat[:, T]

    # Step. 'none' keeps the original scalar-per-voxel eps = step*sigma; 'diag'
    # equalizes the prior and likelihood curvatures (see _diag_precond).
    if precondition == 'diag':
        M = _diag_precond(sigma, source_img, gamma, guidance_scale)
    elif precondition == 'none':
        M = sigma.unsqueeze(1)
    else:
        raise ValueError(f"Unknown precondition: {precondition!r}")
    # Standard overdamped-Langevin discretization:
    #     v <- v + eps * grad log p + sqrt(2 eps) z
    # eps = step_size * M, where M is the preconditioner (sigma, or the diagonal
    # preconditioner) and step_size is the dimensionless stability margin: ULA on
    # a Gaussian of variance s is stable only for eps < 2s, and the bridge
    # transition variance is sigma*rho_t <= sigma, so step_size must stay < 1.
    if scale_step_with_sigma:
        eps = step_size * M                      # [B, 1, C, H, W]
    else:
        # Unscaled ULA: eps is an absolute step, the same for every voxel and
        # frame, with no dependence on sigma. step_size is then NOT the
        # dimensionless margin -- stability still needs eps < 2 sigma rho_t, so
        # a value tuned for the scaled mode will diverge here.
        if precondition == 'diag':
            raise ValueError(
                "scale_step_with_sigma=False is incompatible with "
                "precondition='diag': the diagonal preconditioner IS the "
                "scaling being switched off. Use precondition='none'.")
        M = torch.ones_like(M)
        eps = torch.full_like(M, float(step_size))
    sqrt_2eps = torch.sqrt(torch.clamp(2.0 * eps, min=0.0))

    # K preconditioning. The preconditioner is A = sigma^(1/2) K sigma^(1/2) and
    # the update is
    #
    #     eta <- eta + eps * A grad log p + sqrt(2 eps) * A^(1/2) z
    #
    # The drift goes through _apply_A (S on both sides of the blur at width w);
    # the noise is sqrt(2 eps) * S K^(1/2) z, i.e. sqrt_2eps already supplies S
    # and only K^(1/2) -- the Gaussian of width w/sqrt(2), since
    # G_a * G_a = G_{a sqrt 2} -- is applied to z. A = (S K^(1/2))(K^(1/2) S), so
    # Cov(noise) = 2 eps A matches the drift operator exactly. That is the
    # property the earlier versions broke: first by blurring only z, which left
    # the drift isotropic, then by using S^2 K for the drift, which is asymmetric
    # wherever sigma varies -- both left the chain targeting nothing well defined.
    k_half_width = (noise_smooth_width / math.sqrt(2.0)
                    if noise_smooth_width and noise_smooth_width > 0 else 0.0)

    # renorm_K: undo the magnitude K removes, WITHOUT breaking the pairing.
    #
    # A blur shrinks a white field's std by a = sum_i k_i^2 (2-D separable), so
    # K^(1/2) z comes out a-times smaller than z and K g roughly a^2 smaller than
    # g. Scaling them back must respect Cov(noise) = 2 eps * (drift operator):
    # multiplying the NOISE by 1/a forces the DRIFT to take (1/a)^2, which is
    # exactly the pair a rescaled preconditioner K' = K/a^2 requires. Restoring
    # each term to its own pre-blur scale therefore happens to be the consistent
    # choice -- the chain still targets p.
    #
    # WHAT IT COSTS. K' = K/a^2 is the same chain as K at step_size/a^2: verified
    # bit-identical (max|diff| 1.2e-7). So this multiplies the EFFECTIVE step by
    # 1/a^2 -- about 1.15x at w=0.5 and 6.1x at w=1, but ~100x at w=4, and the
    # sampler diverges somewhere between step_size 0.4 and 1.0. Lower
    # --langevin_step_size to match, or it will blow up at the wider kernels.
    c_noise = c_drift = 1.0
    if renorm_K and k_half_width > 0:
        _k, _ = _gauss_kernel_1d(k_half_width, sigma.device, sigma.dtype)
        _a = float((_k * _k).sum())
        c_noise = 1.0 / max(_a, 1e-12)
        c_drift = c_noise ** 2

    def _grad_logp(e):
        """(-grad of -log p) and, for MALA, the per-chain log p."""
        e = e.detach().requires_grad_(True)
        lp = _logp_per_chain(e, v_hat, sigma, source_img, target_imgs, inshape,
                             gamma, guidance_scale, reg_weight, T, smooth_width)
        g = torch.autograd.grad(lp.sum(), e, create_graph=False)[0]
        return lp.detach(), g.detach()

    def _pin(e):
        """Enforce the bridge boundary conditions.

        Default (the paper, Eq. 4): the PERTURBATION is pinned, eta_0 = eta_T = 0,
        so v_0 = v_hat_0 and v_T = v_hat_T.

        pin_v_endpoints=True: pin the VELOCITY instead, v_0 = v_T = 0, i.e.
        eta_T = -v_hat_T. Note this does NOT change S or sigma*: S sums only
        t = 1..T-1 and the t=T transition is excluded as degenerate, so eta_T
        never enters it. It changes the displayed trajectory and adds a constant
        to the energy at t=T (whose gradient w.r.t. the free eta_1..eta_{T-1}
        is zero).
        """
        e[:, 0] = -v_hat[:, 0]
        e[:, T] = -v_hat[:, T] if pin_v_endpoints else torch.zeros_like(e[:, T])
        return e

    n_acc, n_prop = 0, 0
    lp, g = _grad_logp(eta) if mala else (None, None)

    n_track = min(int(grad_steps), int(n_steps)) if grad_steps else 0
    first_tracked = int(n_steps) - n_track

    # collect_every > 0 snapshots the trajectory every k-th step DURING the walk
    # instead of only at the end. That is what lets one call produce many
    # samples whose gradients all trace back through the SAME chain: taking them
    # as separate calls detaches between segments and throttles dL/dsigma by
    # ~10x (measured 0.35 vs 3.37 for ten one-step samples).
    collected = [] if collect_every and collect_every > 0 else None

    def _snapshot(e_state):
        """Velocity trajectory at the current chain state, graph intact."""
        if smooth_width and smooth_width > 0:
            B_, Tp1 = e_state.shape[0], e_state.shape[1]
            sm = smooth_field(e_state.reshape(B_ * Tp1, *e_state.shape[2:]),
                              smooth_width)
            sm = sm.reshape(B_, Tp1, *e_state.shape[2:])
        else:
            sm = e_state
        vv = v_hat + sm
        return [vv[:, t] for t in range(T + 1)]

    for k in range(int(n_steps)):
        if not mala:
            e_in = eta.detach().requires_grad_(True)
            lp_k = _logp_per_chain(e_in, v_hat, sigma.detach(), source_img,
                                   target_imgs, inshape, gamma, guidance_scale,
                                   reg_weight, T, smooth_width)
            g = torch.autograd.grad(lp_k.sum(), e_in, create_graph=False)[0]
            z = torch.randn(eta.shape, device=eta.device, dtype=eta.dtype,
                            generator=generator)

            def _step(e, gg, zz, _eps=eps, _s2e=sqrt_2eps):
                """One preconditioned ULA step: drift and noise share A = S K S.

                K off collapses to the plain eps*g + sqrt(2 eps)*z, since
                S K S with K = I is exactly diag(eps) -- the default path is
                unchanged.

                The prior draw that seeds the chain stays WHITE: it is an exact
                bridge-prior sample, and the preconditioner shapes the walk, not
                the target, so the initialization needs no matching blur.
                """
                if k_half_width > 0:
                    # Drift: eps A g with A = S K S. S sits on BOTH sides of K,
                    # so the drift operator equals Cov(noise)/2 exactly. Passing
                    # _eps as _apply_A's sigma folds the step size in, giving
                    # sqrt(eps) K sqrt(eps) g = step_size * S K S g.
                    _ag = _apply_A(gg, _eps, noise_smooth_width) * c_drift
                    # Noise: sqrt(2 eps) A^(1/2) z, realized as sqrt(2) S K^(1/2) z
                    # since A = (S K^(1/2))(K^(1/2) S). _s2e = sqrt(2 eps) already
                    # carries S, so only K^(1/2) is applied here -- a Gaussian of
                    # width w/sqrt(2), since G_a * G_a = G_{a sqrt2}.
                    _kz = _apply_K(zz, k_half_width) * c_noise
                    if grad_bypass_K:
                        # Straight-through: the VALUE is the smoothed step, but
                        # d(step)/d(eps) sees the UNSMOOTHED field, so the
                        # pathwise dL/dsigma keeps the energy gradient K would
                        # otherwise eat (K suppresses grad E to 3.0% against
                        # 9.1% for the prior term, diluting the image signal).
                        # x*a + (x*b - x*a).detach()-style: value b, slope a.
                        drift = _eps * gg + (_ag - _eps * gg).detach()
                        noise = _s2e * zz + (_s2e * _kz - _s2e * zz).detach()
                    else:
                        drift = _ag
                        noise = _s2e * _kz
                else:
                    drift = _eps * gg
                    noise = _s2e * zz
                return _pin(e + drift + noise)

            if k >= first_tracked and n_track:
                # Pathwise step: the SCORE is detached (as the reverse SDE
                # detaches grad_E) but S and sqrt(2 eps) carry sigma, so
                # d eta / d sigma survives -- S appears TWICE in the drift under
                # K, once on each side, and once inside sqrt_2eps. With the noise
                # z fixed this is the reparameterization gradient, the same
                # construction that makes the reverse SDE differentiable w.r.t.
                # its scaling factors. Under scale_step_with_sigma=False eps and
                # S are constants, so this path contributes exactly zero
                # dL/dsigma -- correct for that update, but it makes grad_steps a
                # no-op.
                eta = _step(eta, g, z)
            else:
                with torch.no_grad():
                    eta = _step(eta.detach(), g, z, eps.detach(),
                                sqrt_2eps.detach())
        else:
            with torch.no_grad():
                mean = _pin(eta + eps * g)
                z = torch.randn(eta.shape, device=eta.device, dtype=eta.dtype,
                                generator=generator)
                prop = _pin(mean + sqrt_2eps * z)
            lp_p, g_p = _grad_logp(prop)
            with torch.no_grad():
                mean_rev = _pin(prop + eps * g_p)
                # log q(.|.) quadratic parts; the shared det(eps) cancels.
                lq_fwd = -(((prop - mean) ** 2) / (4 * eps)).flatten(1).sum(1)
                lq_rev = -(((eta - mean_rev) ** 2) / (4 * eps)).flatten(1).sum(1)
                log_alpha = (lp_p - lp) + (lq_rev - lq_fwd)
                u = torch.rand(log_alpha.shape, device=log_alpha.device,
                               generator=generator)
                acc = torch.log(u.clamp_min(1e-30)) < log_alpha        # [B]
                sel = acc.view(-1, *([1] * (eta.dim() - 1)))
                eta = torch.where(sel, prop, eta)
                lp = torch.where(acc, lp_p, lp)
                g = torch.where(sel, g_p, g)
                n_acc += int(acc.sum()); n_prop += int(acc.numel())

        if collected is not None and (k + 1) % int(collect_every) == 0:
            collected.append(_snapshot(eta))

        if torch.isnan(eta).any():
            raise RuntimeError(
                f"NaN in Langevin sampler at step {k}. eps "
                f"({'step_size*M' if scale_step_with_sigma else 'step_size, unscaled'}) "
                f"is too large, or gamma too small. step_size={step_size:g}, "
                f"gamma={gamma:g}, precondition={precondition}, mala={mala}, "
                f"scale_step_with_sigma={scale_step_with_sigma}, "
                f"noise_smooth_width={noise_smooth_width:g}. "
                f"With precondition='diag' the likelihood curvature is already "
                f"absorbed, so lower step_size rather than raising gamma. "
                f"With scale_step_with_sigma=False step_size is an absolute "
                f"step and must be on the scale of sigma "
                f"(sigma in [{float(sigma.min()):.3g}, {float(sigma.max()):.3g}] "
                f"here), not the 0.4-ish value the scaled mode uses.")

    if mala:
        MALA_STATS['acc'] += n_acc
        MALA_STATS['prop'] += n_prop

    out_ctx = _NoGrad() if not n_track else _KeepGrad()
    with out_ctx:
        if smooth_width and smooth_width > 0:
            B_, Tp1 = eta.shape[0], eta.shape[1]
            e = smooth_field(eta.reshape(B_ * Tp1, *eta.shape[2:]), smooth_width)
            e = e.reshape(B_, Tp1, *eta.shape[2:])
        else:
            e = eta
        v = v_hat + e
        v_t_list = [v[:, t].clone() for t in range(T + 1)]
        std = [torch.zeros_like(v_t_list[0]) for _ in range(T + 1)]
        # The LATENT xi is what the closed-form M-step is defined on. It is
        # always detached: it seeds the NEXT chain segment, and carrying a graph
        # across segments would chain them without bound.
        xi_list = [eta[:, t].detach().clone() for t in range(T + 1)]
    if collected is not None:
        # trailing element; the other variants are unchanged for existing callers
        if return_latent:
            return v_t_list, std, xi_list, collected
        return v_t_list, std, collected
    if return_latent:
        if return_stats:
            return v_t_list, std, xi_list, {'accept_rate': (n_acc / n_prop) if n_prop else float('nan'),
                                            'eps_min': float(eps.min()), 'eps_max': float(eps.max())}
        return v_t_list, std, xi_list
    if return_stats:
        return v_t_list, std, {'accept_rate': (n_acc / n_prop) if n_prop else float('nan'),
                               'eps_min': float(eps.min()), 'eps_max': float(eps.max())}
    return v_t_list, std


def langevin_chain(v_reg_list, sigma, source_img, target_imgs,
                   n_burn=1000, n_collect=200, thin=1, keep=4,
                   generator=None, verbose=False, **kw):
    """
    Run ONE chain and use every post-burn-in state as a sample.

    `langevin_velocities` returns a single state -- the end of the walk -- which
    is what the trainer needs, since compute_brownian_bridge_velocities hands
    back one trajectory per call. That is the wrong shape for estimating a
    posterior, because it discards the n_collect-1 states the chain passed
    through on the way. Standard MCMC keeps them: once the chain is burned in,
    EVERY subsequent state is a draw from the stationary distribution, so a run
    of B + N steps yields N samples, not one.

    This function does that. It burns in for n_burn steps, then advances `thin`
    steps at a time and treats each resulting state as a sample.

    Mean and variance are accumulated online (Welford), so memory is O(1) in
    n_collect rather than O(n_collect * B * T * H * W) -- at B=8, T=14, 64x64,
    storing 200 raw samples would be ~2 GB. `keep` full samples are retained for
    display.

    Correlation. Successive states are correlated (the slowest bridge mode has a
    relaxation time of ~400 steps at step_size 0.05), so N states are worth
    fewer than N independent draws and the reported std carries more error than
    an i.i.d. estimate of the same count would. It is still a consistent
    estimator of the stationary variance -- an ergodic average -- and strictly
    better than the single state it replaces. Raise `thin` to decorrelate, or
    run several chains in the batch axis (they are independent) and combine.

    Returns:
        dict with 'mean' and 'std' (lists of T+1 tensors [B,C,H,W]),
        'samples' (list of `keep` v_t_lists), 'xi' (final chain state, for
        resuming), and 'n' (number of samples averaged).
    """
    kw.pop('return_latent', None)
    kw.pop('return_stats', None)
    kw.pop('n_steps', None)
    kw.pop('init_eta', None)

    _, _, xi = langevin_velocities(
        v_reg_list, sigma, source_img, target_imgs, n_steps=n_burn,
        init='prior', generator=generator, return_latent=True, **kw)
    if verbose:
        print(f"[langevin_chain] burn-in {n_burn} done; collecting "
              f"{n_collect} samples every {thin} step(s)")

    mean = None       # running mean, [T+1] list of tensors
    m2 = None         # running sum of squared deviations
    samples = []
    for i in range(int(n_collect)):
        v_t_list, _, xi = langevin_velocities(
            v_reg_list, sigma, source_img, target_imgs, n_steps=thin,
            init_eta=xi, generator=generator, return_latent=True, **kw)
        with torch.no_grad():
            if mean is None:
                mean = [v.clone() for v in v_t_list]
                m2 = [torch.zeros_like(v) for v in v_t_list]
            else:
                n = i + 1
                for t, v in enumerate(v_t_list):
                    d = v - mean[t]
                    mean[t] += d / n
                    m2[t] += d * (v - mean[t])
            if len(samples) < keep:
                samples.append([v.clone() for v in v_t_list])
        if verbose and (i + 1) % max(1, n_collect // 10) == 0:
            print(f"[langevin_chain]   {i + 1}/{n_collect} "
                  f"(chain step {n_burn + (i + 1) * thin})")

    n = int(n_collect)
    std = [torch.sqrt(x / max(n - 1, 1)) for x in m2]
    return {'mean': mean, 'std': std, 'samples': samples, 'xi': xi, 'n': n}


def draw_velocities(sampler, trainer, v_reg_list, scaling_factors_bb,
                    source_img, target_imgs, num_samples=1, sigma=None,
                    langevin_kw=None):
    """
    Dispatch between the Eq. 14 reverse SDE and the Langevin sampler.

    `sigma` is required for 'langevin' and must already be tiled to match the
    batch of `v_reg_list`.
    """
    if sampler == 'reverse_sde':
        return trainer.compute_brownian_bridge_velocities(
            v_reg_list, scaling_factors_bb,
            source_img=source_img, target_imgs=target_imgs,
            num_samples=num_samples,
        )
    if sampler != 'langevin':
        raise ValueError(f"Unknown sampler: {sampler!r}")
    if sigma is None:
        raise ValueError("sampler='langevin' needs sigma (same units as the M-step)")
    kw = dict(langevin_kw or {})
    kw.setdefault('gamma', trainer.gamma)
    kw.setdefault('guidance_scale', trainer.guidance_scale)
    kw.setdefault('reg_weight', getattr(trainer.brownian_bridge, 'reg_weight', 0.1))
    kw.setdefault('inshape', (trainer.img_size, trainer.img_size))
    kw.pop('return_stats', None)
    kw.setdefault('pin_v_endpoints', False)
    return langevin_velocities(v_reg_list, sigma, source_img, target_imgs, **kw)
