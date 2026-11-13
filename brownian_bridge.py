"""
Brownian Bridge Diffusion with learned spatially-varying scaling factors
and Posterior Reverse SDE for uncertainty quantification.

Identical to uncertainty_inv_gamma_prior/brownian_bridge.py except:
- run_posterior_reverse_process: adds energy gradient correction to the
  discrete bridge formula to steer samples toward lower registration error.

The posterior correction modifies the bridge conditional mean:
    mu_posterior = mu_tilde - (sigma^2 / gamma) * grad_E(v_t)

where sigma^2 = 2s/T, and grad_E is detached (only steers samples,
does not affect training gradients which flow through mu_tilde and std_tilde).
"""

import os
import sys

import torch
import torch.nn.functional as F
import numpy as np

from warping import warp_image_with_velocity
from losses import Grad


class BrownianBridgeLearnedScaling:
    """
    Brownian Bridge Diffusion with learned spatially-varying scaling factors
    and posterior reverse SDE support.
    """

    def __init__(self, num_diffusion_steps=1000, img_size=64, reg_weight=0.1,
                 analytic_reg_grad=False, antithetic=False, energy_frame_chunk=0):
        self.T = num_diffusion_steps
        self.m = torch.linspace(0, 1, self.T + 1)
        self.img_size = img_size
        self.grad_loss = Grad(penalty='l2')
        self.reg_weight = reg_weight
        # Use the closed-form gradient of the l2 smoothness penalty inside
        # compute_energy_gradient instead of routing it through autograd.
        # Only valid for penalty='l2' (see _analytic_grad_l2).
        self.analytic_reg_grad = bool(analytic_reg_grad) and self.grad_loss.penalty == 'l2'
        # Draw the reverse-SDE noise in antithetic pairs (z, -z). See _draw_noise.
        self.antithetic = bool(antithetic)
        # How many of the T-1 frames in the Eq. 13 sum to stack into one
        # warp+backward inside compute_energy_gradient. 0 = all at once.
        # Purely a memory knob: gradients of a sum are additive, so chunking
        # is numerically identical, just slower and lighter.
        self.energy_frame_chunk = int(energy_frame_chunk)

    def compute_delta_t(self, t, scaling_factors):
        """Compute delta_t = 2s(m_t - m_t^2) with spatially-varying s."""
        m_t = self.m[t]
        return 2 * scaling_factors * (m_t - m_t ** 2)

    def compute_delta_conditional(self, t, scaling_factors_t, scaling_factors_t_minus_1):
        """Compute delta_{t|t-1} with spatially-varying scaling factors."""
        if t <= 1:
            return torch.zeros_like(scaling_factors_t)

        delta_t = self.compute_delta_t(t, scaling_factors_t)
        delta_t_minus_1 = self.compute_delta_t(t - 1, scaling_factors_t_minus_1)

        m_t = self.m[t]
        m_t_minus_1 = self.m[t - 1]

        ratio = ((1 - m_t) / (1 - m_t_minus_1 + 1e-10)) ** 2
        delta_conditional = delta_t - delta_t_minus_1 * ratio

        return torch.clamp(delta_conditional, min=1e-10)

    def compute_coefficients(self, t, scaling_factors_t, scaling_factors_t_minus_1):
        """Compute coefficients c_xt, c_x0, c_y for the reverse process mean."""
        if t <= 1:
            ones = torch.ones_like(scaling_factors_t)
            zeros = torch.zeros_like(scaling_factors_t)
            return zeros, ones, zeros, zeros

        m_t = self.m[t]
        m_t_minus_1 = self.m[t - 1]

        delta_t = self.compute_delta_t(t, scaling_factors_t)
        delta_t_minus_1 = self.compute_delta_t(t - 1, scaling_factors_t_minus_1)
        delta_conditional = self.compute_delta_conditional(t, scaling_factors_t, scaling_factors_t_minus_1)

        delta_t_safe = torch.clamp(delta_t, min=1e-10)

        alpha = (1 - m_t) / (1 - m_t_minus_1 + 1e-10)
        beta = m_t - alpha * m_t_minus_1

        c_xt = (delta_t_minus_1 / delta_t_safe) * alpha
        c_x0 = (delta_conditional * (1 - m_t_minus_1)) / delta_t_safe
        c_y = (delta_conditional * m_t_minus_1 - delta_t_minus_1 * alpha * beta) / delta_t_safe

        delta_tilde = (delta_conditional * delta_t_minus_1) / delta_t_safe
        delta_tilde = torch.clamp(delta_tilde, min=0.0)

        return c_xt, c_x0, c_y, delta_tilde

    def interpolate_velocity(self, v_reg_list, cardiac_frame_idx):
        """Interpolate velocity field for a given diffusion timestep."""
        K = len(v_reg_list)
        cardiac_frame_idx = max(0, min(K - 1, cardiac_frame_idx))

        idx_low = int(np.floor(cardiac_frame_idx))
        idx_high = int(np.ceil(cardiac_frame_idx))

        if idx_low == idx_high or idx_high >= K:
            return v_reg_list[min(idx_low, K - 1)]

        alpha = cardiac_frame_idx - idx_low
        return v_reg_list[idx_low] * (1 - alpha) + v_reg_list[idx_high] * alpha

    def interpolate_scaling_factors(self, scaling_factors, cardiac_frame_idx):
        """Interpolate scaling factors for a given cardiac frame index."""
        T = scaling_factors.shape[1]
        cardiac_frame_idx = max(0, min(T - 1, cardiac_frame_idx))

        idx_low = int(np.floor(cardiac_frame_idx))
        idx_high = int(np.ceil(cardiac_frame_idx))

        if idx_low == idx_high or idx_high >= T:
            return scaling_factors[:, min(idx_low, T - 1)]

        alpha = cardiac_frame_idx - idx_low
        return scaling_factors[:, idx_low] * (1 - alpha) + scaling_factors[:, idx_high] * alpha

    def diffusion_timestep_to_cardiac_frame(self, t, num_cardiac_frames):
        """Map diffusion timestep t to cardiac frame index."""
        K = num_cardiac_frames - 1
        return K * t / self.T

    def _draw_noise(self, v_t, num_samples, device=None):
        """
        Reverse-SDE noise for a [num_samples*B, 2, H, W] batch.

        Plain: one i.i.d. standard normal per element.

        Antithetic: draw noise for the first ceil(N/2) sample blocks and use its
        negation for the rest. The trainer tiles inputs with
        `t.repeat(N, ...)`, so the batch axis is laid out sample-major —
        index n*B + b — and reshaping to [N, B, ...] puts sample block n on
        axis 0 with the subject index b aligned across blocks. Pairing block n
        with block n + half therefore pairs two trajectories of the *same*
        subject, which is what makes their errors cancel in the per-subject
        mean over N.

        Each individual trajectory keeps its exact marginal law, because -z is
        standard normal whenever z is. The N-sample mean is therefore still
        unbiased; only its variance changes. How much it drops depends on how
        close the functional is to linear in the noise — the h-transform, score
        and sigma*z terms are linear, but grad_E is not, so the reduction is
        not guaranteed to be the full 2x and has to be measured, not assumed.

        Falls back to plain i.i.d. noise when the batch does not split evenly
        into N blocks, or when N < 2.
        """
        device = device if device is not None else v_t.device
        N = int(num_samples)
        total = v_t.shape[0]

        if not self.antithetic or N < 2 or total % N != 0:
            return torch.randn_like(v_t, device=device)

        B = total // N
        half = (N + 1) // 2                       # ceil, so odd N keeps one unpaired
        rest = v_t.shape[1:]

        z_half = torch.randn(half, B, *rest, dtype=v_t.dtype, device=device)
        z = torch.cat([z_half, -z_half[:N - half]], dim=0)
        return z.reshape(total, *rest)

    @staticmethod
    def _analytic_grad_l2(v):
        """
        Closed-form gradient of losses.Grad(penalty='l2').loss(v) w.r.t. v.

        The penalty is
            R(v) = (mean(dx^2) + mean(dy^2)) / 2
        with the *forward differences*
            dy = v[:, :, 1:, :] - v[:, :, :-1, :]   (N_y = B*C*(H-1)*W elements)
            dx = v[:, :, :, 1:] - v[:, :, :, :-1]   (N_x = B*C*H*(W-1) elements)
        (the torch.abs in Grad.loss is a no-op once squared).

        dR/d(dy_j) = dy_j / N_y, and each dy_j = v[k] - v[k-1] contributes +1 to
        v[k] and -1 to v[k-1], so scattering the differences back gives

            dR/dv[k] = (2*v[k] - v[k-1] - v[k+1]) / N_y   (interior, y-direction)

        i.e. the negative discrete Laplacian, with one-sided stencils at the
        boundaries — which the scatter below produces automatically. Note N_y and
        N_x differ (the two difference tensors have different sizes), so the two
        directions carry different normalizations; this reproduces Grad.loss
        exactly rather than approximating it.
        """
        if v.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got shape {tuple(v.shape)}")

        dy = v[:, :, 1:, :] - v[:, :, :-1, :]
        dx = v[:, :, :, 1:] - v[:, :, :, :-1]
        dy = dy / dy.numel()
        dx = dx / dx.numel()

        g = torch.zeros_like(v)
        g[:, :, 1:, :] += dy
        g[:, :, :-1, :] -= dy
        g[:, :, :, 1:] += dx
        g[:, :, :, :-1] -= dx
        return g

    def _bridge_coefficient(self, s, paper_t):
        """
        c_{s,t} of Eq. 13: the sensitivity of the bridge conditional mean at
        frame s to the perturbation at the current frame t,

            c_{s,t} = s / t            for s <= t
                    = (T - s)/(T - t)  for s >  t
        """
        if s <= paper_t:
            return s / max(paper_t, 1e-10)
        return (self.T - s) / max(self.T - paper_t, 1e-10)

    def compute_energy_gradient(self, v_t, v_reg_list, source_img, target_imgs,
                                paper_t, num_cardiac_frames):
        """
        Likelihood score of Eq. 13 (unscaled), detached.

            sum_{s=1}^{T-1} grad_{v_t} E(mu_{s|t}),
                mu_{s|t} = v_hat_s + c_{s,t} * eta_t,   eta_t = v_t - v_hat_t
                c_{s,t}  = s/t (s <= t),  (T-s)/(T-t) (s > t)

        The likelihood in Eq. 8 marginalizes the velocity at every other
        timepoint, which is intractable; following Chung et al. (2023) it is
        approximated by evaluating the registration energy at the Brownian
        bridge conditional means mu_{s|t} of all frames s, not just at the
        current frame. Each mu_{s|t} is affine in v_t, so a single
        autograd.grad over the summed energy already produces
        sum_s c_{s,t} * dE/dmu|_{mu_{s|t}} by the chain rule.

        Used for posterior SDE steering only, not for training gradients.
        """
        if self.T < 2:
            return torch.zeros_like(v_t)

        inshape = (self.img_size, self.img_size)

        # Frames s = 1 ... T-1 of the sum, with their bridge coefficient and
        # the image they are matched against (target_imgs[j] is frame j+1).
        frames = []
        for s in range(1, self.T):
            cardiac_idx_s = self.diffusion_timestep_to_cardiac_frame(s, num_cardiac_frames)
            target_idx = int(round(cardiac_idx_s)) - 1
            if target_idx < 0 or target_idx >= len(target_imgs):
                continue
            frames.append((cardiac_idx_s, self._bridge_coefficient(s, paper_t), target_idx))

        if not frames:
            return torch.zeros_like(v_t)

        v_t_grad = v_t.detach().clone().requires_grad_(True)
        cardiac_idx_t = self.diffusion_timestep_to_cardiac_frame(paper_t, num_cardiac_frames)
        v_hat_t = self.interpolate_velocity(v_reg_list, cardiac_idx_t).detach()
        eta_t = v_t_grad - v_hat_t

        chunk = self.energy_frame_chunk if self.energy_frame_chunk > 0 else len(frames)
        rep = (1,) * (source_img.dim() - 1)

        grad_E = torch.zeros_like(v_t_grad)

        for start in range(0, len(frames), chunk):
            block = frames[start:start + chunk]
            S = len(block)

            # mu_{s|t} for every frame in this block, stacked along the batch
            # axis so the whole block costs one warp instead of S warps.
            mus = [self.interpolate_velocity(v_reg_list, idx).detach() + c * eta_t
                   for idx, c, _ in block]
            mu_all = torch.cat(mus, dim=0)
            target_all = torch.cat([target_imgs[j] for _, _, j in block], dim=0)
            source_all = source_img.repeat(S, *rep)

            warped = warp_image_with_velocity(source_all, mu_all, inshape=inshape)
            # mse_loss averages over the whole stack; multiplying by S turns the
            # mean of the S per-frame means back into their sum.
            energy = F.mse_loss(warped, target_all) * S

            if self.analytic_reg_grad:
                # Only the MSE term goes through autograd (its gradient w.r.t. v
                # needs the adjoint of VecInt + grid_sample, which has no closed
                # form). The smoothness term is a fixed linear stencil, so its
                # gradient is computed directly and never enters the graph --
                # which means the chain-rule factor c_{s,t} has to be applied by
                # hand here, unlike on the autograd path.
                g = torch.autograd.grad(energy, v_t_grad, create_graph=False)[0]
                if self.reg_weight != 0:
                    for mu, (_, c, _) in zip(mus, block):
                        g = g + self.reg_weight * c * self._analytic_grad_l2(mu.detach())
            else:
                if self.reg_weight != 0:
                    energy = energy + self.reg_weight * self.grad_loss.loss(mu_all) * S
                g = torch.autograd.grad(energy, v_t_grad, create_graph=False)[0]

            grad_E = grad_E + g

            del mus, mu_all, target_all, source_all, warped, energy, g

        return grad_E.detach()

    def run_reverse_process(self, v_reg_list, scaling_factors, v_0, v_T, device='cuda',
                            num_samples=1):
        """
        Run standard (prior) reverse diffusion process. Identical to inv_gamma_prior.

        Uses discrete bridge formula: v_t = mu_tilde + std_tilde * z
        """
        num_cardiac_frames = len(v_reg_list)

        v_t = v_T.clone()
        trajectory = [v_t.clone()]

        for t in range(self.T - 1, -1, -1):
            if t == 0:
                v_t = v_0.clone()
            else:
                cardiac_idx_t = self.diffusion_timestep_to_cardiac_frame(t, num_cardiac_frames)
                cardiac_idx_t_plus_1 = self.diffusion_timestep_to_cardiac_frame(t + 1, num_cardiac_frames)

                s_t = self.interpolate_scaling_factors(scaling_factors, cardiac_idx_t)
                s_t_plus_1 = self.interpolate_scaling_factors(scaling_factors, cardiac_idx_t_plus_1)

                c_xt, c_x0, c_y, delta_tilde = self.compute_coefficients(t + 1, s_t_plus_1, s_t)

                v_t_reg = self.interpolate_velocity(v_reg_list, cardiac_idx_t_plus_1)
                v_t_minus_1_reg = self.interpolate_velocity(v_reg_list, cardiac_idx_t)

                if t > 0:
                    z = self._draw_noise(v_t, num_samples, device=device)
                else:
                    z = torch.zeros_like(v_t, device=device)

                mu_tilde = (c_xt * v_t + c_x0 * v_0 + c_y * v_T +
                           v_t_minus_1_reg - c_xt * v_t_reg)
                std_tilde = torch.sqrt(torch.clamp(delta_tilde, min=1e-10))
                v_t = mu_tilde + std_tilde * z

                del z, mu_tilde

            trajectory.append(v_t.clone())

        return trajectory

    def run_posterior_reverse_process(self, v_reg_list, scaling_factors, v_0, v_T,
                                      source_img, target_imgs,
                                      gamma=1.0, guidance_scale=1.0, device='cuda',
                                      num_samples=1):
        """
        Posterior conditional reverse SDE (Eq. 14 of the paper).

        Euler-Maruyama discretization of Eq. 11 with Delta_t = -1:
            v_{t-1} = v_t + (v^r_{t-1} - v^r_t)                [Delta v_hat_t]
                      + eta_t/(T-t)                            [h-transform, f_BB]
                      - (sigma/gamma) * sum_{s=1}^{T-1}
                            grad_{v_t} E(mu_{s|t})             [likelihood score, Eq. 13]
                      - sigma * eta_t / delta_t                [bridge prior score]
                      + sqrt(sigma) * z                        [noise]

        where eta_t = v_t - v^r_t and the paper's diffusion variance sigma^i is
        `sigma_sq = 2s/T` below, so delta_t = sigma * t * (1 - t/T)
        = 2s * m_t * (1-m_t) and the noise scale is sqrt(sigma_sq).

        The likelihood term is the full Eq. 13 sum over all frames, evaluated
        at the bridge conditional means mu_{s|t} = v_hat_s + c_{s,t} * eta_t
        (see compute_energy_gradient), not just at the current frame.

        This is a direct computation (no accumulator), producing variance
        that peaks at the midpoint t=T/2 as expected from the bridge prior.
        """
        num_cardiac_frames = len(v_reg_list)

        v_t = v_T.clone()
        trajectory = [v_t.clone()]

        for t in range(self.T - 1, -1, -1):
            if t == 0:
                v_t = v_0.clone()
            else:
                # paper_t = t+1: the diffusion step we are currently at
                paper_t = t + 1

                cardiac_idx_current = self.diffusion_timestep_to_cardiac_frame(paper_t, num_cardiac_frames)
                cardiac_idx_prev = self.diffusion_timestep_to_cardiac_frame(t, num_cardiac_frames)

                # Registered velocities
                v_t_reg = self.interpolate_velocity(v_reg_list, cardiac_idx_current)
                v_t_minus_1_reg = self.interpolate_velocity(v_reg_list, cardiac_idx_prev)

                # Perturbation at current step
                eta_t = v_t - v_t_reg

                # sigma^2 = 2s/T
                # Force fp32 here even if caller is under autocast: s_current is
                # clamped to [1e-4, 1e4] so sigma_sq can hit ~1.4e-5 (fp16
                # subnormal) and delta_t ~1e-6 — fp16 division in `score` would
                # underflow. Coefficients stay fp32; the additive update at the
                # bottom inherits v_t's dtype via promotion.
                s_current = self.interpolate_scaling_factors(scaling_factors, cardiac_idx_current).float()
                sigma_sq = 2.0 * s_current / self.T
                sigma = torch.sqrt(torch.clamp(sigma_sq, min=1e-10))

                # delta_t = sigma^2 * paper_t * (1 - paper_t/T) (bridge marginal variance)
                m_t = paper_t / self.T
                delta_t = sigma_sq * paper_t * (1.0 - m_t)
                delta_t = torch.clamp(delta_t, min=1e-10)

                # Eq. 13 likelihood score, summed over all frames (detached —
                # steers samples, no training gradients)
                grad_E = self.compute_energy_gradient(
                    v_t, v_reg_list, source_img, target_imgs,
                    paper_t, num_cardiac_frames
                )

                z = self._draw_noise(v_t, num_samples, device=device)

                # Eq. 14 three drift terms. Coefficient math is fp32 (above);
                # cast each term to v_t.dtype before summing so v_t stays in its
                # original dtype (fp16 under autocast) — preserves the memory
                # win without losing the fp32 protection on the small-scale
                # divides.
                v_dtype = v_t.dtype
                # 1. h-transform (bridge drift f_BB, sign flipped by Delta_t=-1)
                h_transform = eta_t / (self.T - paper_t + 1e-10)
                # 2. energy gradient correction (fp32 score, then cast)
                energy_correction = (-(guidance_scale / gamma) * sigma_sq * grad_E.float()).to(v_dtype)
                # 3. marginal score: -sigma^2 * eta_t / delta_t (fp32 score, then cast)
                score = (-sigma_sq * eta_t.float() / delta_t).to(v_dtype)
                sigma_z = (sigma * z.float()).to(v_dtype)

                v_t = (v_t + (v_t_minus_1_reg - v_t_reg)
                       + h_transform + energy_correction + score + sigma_z)

                if torch.isnan(v_t).any():
                    print(f"\n  [FATAL] NaN detected in v_t at reverse step t={t}")
                    raise RuntimeError(f"NaN detected at reverse step t={t}")

            trajectory.append(v_t.clone())

        return trajectory

    def run_posterior_forward_process(self, v_reg_list, scaling_factors, v_0, v_T,
                                      source_img, target_imgs,
                                      gamma=1.0, guidance_scale=1.0, device='cuda'):
        """
        Solve the posterior reverse SDE using forward Euler-Maruyama (dt=+1).

        Same SDE as run_posterior_reverse_process but discretized in the
        forward direction: starting from v_0 (t=0) and stepping to v_T (t=T).

        Reverse EM (current code) uses mu_posterior = -b (negated drift), then:
            v_tilde += mu_posterior + sigma*z      (dt = -1 absorbed)

        Forward EM flips the sign:
            v_tilde += -mu_posterior + sigma*z     (dt = +1)
            v_tilde += (+score + h_transform + correction) + sigma*z

        Three drift terms (with forward signs):
        1. Score:       +sigma^2 * (v_tilde - mu) / delta_t
        2. h-transform: +(y - v_tilde) / (T - t)
        3. Energy grad: +(sigma^2 / gamma) * grad_E
        """
        num_cardiac_frames = len(v_reg_list)

        v_t = v_0.clone()
        v_tilde = torch.zeros_like(v_0)
        trajectory = [v_t.clone()]

        for t in range(1, self.T + 1):
            if t == self.T:
                v_t = v_T.clone()
            else:
                # Map diffusion timesteps to cardiac frames
                cardiac_idx_t_minus_1 = self.diffusion_timestep_to_cardiac_frame(t - 1, num_cardiac_frames)
                cardiac_idx_t = self.diffusion_timestep_to_cardiac_frame(t, num_cardiac_frames)

                # Scaling factors at current position (t-1)
                s_t_minus_1 = self.interpolate_scaling_factors(scaling_factors, cardiac_idx_t_minus_1)

                # Registered velocities
                v_t_minus_1_reg = self.interpolate_velocity(v_reg_list, cardiac_idx_t_minus_1)
                v_t_reg = self.interpolate_velocity(v_reg_list, cardiac_idx_t)

                # sigma^2 = 2s/T at current position (t-1)
                sigma_sq = 2.0 * s_t_minus_1 / self.T

                z = torch.randn_like(v_t, device=device)

                # Marginal variance at t-1: delta = 2*s*m*(1-m)
                delta_t_minus_1 = self.compute_delta_t(t - 1, s_t_minus_1)
                delta_t_minus_1 = torch.clamp(delta_t_minus_1, min=1e-10)

                m_t_minus_1 = (t - 1) / self.T

                # 1. Score: +sigma^2 * (v_tilde - mu) / delta
                #    mu = (1-m)*v_0 + m*v_T (bridge mean in code convention)
                score = sigma_sq * (((v_t - v_t_minus_1_reg) - (1 - m_t_minus_1) * v_0 - m_t_minus_1 * v_T) / delta_t_minus_1)

                # 2. h-transform: +(y - v_tilde) / (T - (t-1))
                h_transform = (v_T - (v_t - v_t_minus_1_reg)) / (self.T - (t - 1))

                # 3. Energy gradient correction (detached)
                grad_E = self.compute_energy_gradient(
                    v_t, v_reg_list, source_img, target_imgs,
                    t - 1, num_cardiac_frames
                )
                correction = (guidance_scale / gamma) * sigma_sq * grad_E

                # Forward EM step: v_tilde += b*dt + sigma*dW, dt = +1
                mu_forward = score + h_transform + correction
                std = torch.sqrt(torch.clamp(sigma_sq, min=1e-10))
                v_tilde = v_tilde + mu_forward + std * z
                v_t = v_t_reg + v_tilde

            trajectory.append(v_t.clone())

        return trajectory

    def sample_bridge_transition(self, v_reg_list, scaling_factors, device='cuda'):
        """
        Sample velocities using the forward bridge transition probability:

            v_t | v_{t-1} ~ N(mu_t, sigma_t^2),   t = 1, ..., T

        where:
            mu_t      = v^r_t + (T - t) / (T - (t-1)) * (v_{t-1} - v^r_{t-1})
            sigma_t^2 = sigma^2(x) * (T - t) / (T - (t-1))

        v_0 is fixed at the boundary. At t = T the ratio is 0, so
        v_T = v^r_T deterministically (bridge endpoint).

        Args:
            v_reg_list: list of K registered velocities [v^r_0, ..., v^r_{K-1}]
            scaling_factors: [B, K, C, H, W] learned scaling factors
            device: torch device

        Returns:
            dict mapping frame_idx -> sampled velocity tensor [B, 2, H, W]
        """
        num_cardiac_frames = len(v_reg_list)
        T = num_cardiac_frames - 1  # last frame index

        sampled = {}
        sampled[0] = v_reg_list[0].clone()  # v_0 fixed (boundary)

        v_prev = sampled[0]

        for t in range(1, num_cardiac_frames):
            ratio = (T - t) / (T - (t - 1))  # (T-1)/T at t=1, ... , 0 at t=T

            s_t = self.interpolate_scaling_factors(scaling_factors, t)  # [B, 2, H, W]

            # Conditional mean
            mean = v_reg_list[t] + ratio * (v_prev - v_reg_list[t - 1])

            if t < T:
                # Conditional variance: sigma^2(x) * ratio
                variance = s_t * ratio
                std = torch.sqrt(torch.clamp(variance, min=1e-10))
                z = torch.randn_like(mean, device=device)
                v_t = mean + std * z
            else:
                # At t = T: ratio = 0 => v_T = v^r_T deterministically
                v_t = v_reg_list[T].clone()

            sampled[t] = v_t.clone()
            v_prev = v_t

        return sampled

    def warp_image(self, image, velocity):
        """Warp an image using a velocity field."""
        inshape = (self.img_size, self.img_size)
        return warp_image_with_velocity(image, velocity, inshape=inshape)
