"""
Trainer for the scaling factor prediction network with posterior reverse SDE.

Key differences from uncertainty_sde_velInput:
  1. The scaling network receives BOTH images AND velocity fields as input.
  2. Uses MLE optimization (num_em_samples=1) instead of EM.
"""

import os
import sys

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from warping import warp_image_with_velocity
from langevin import (
    langevin_velocities, _prior_draw)


class _NullContext:
    """No-op context manager — used in place of autocast when AMP is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class ScalingFactorTrainer:
    """
    Trainer for the scaling factor prediction network.
    Uses posterior reverse SDE and combined image+velocity input scaling network.
    Uses MLE optimization (single sample, no EM averaging).
    """

    def __init__(self, scaling_network, brownian_bridge, loss_fn,
                 lr=1e-4, device='cuda', img_size=64,
                 gamma=1.0, guidance_scale=1.0,
                 use_amp=False, sampler='reverse_sde', langevin_kw=None,
                 weight_decay=1e-5, sigma_optimizer='adam', sigma_momentum=0.9):
        # scaling_network may be None when only compute_brownian_bridge_velocities
        # is used (e.g. test-time inference from a saved s^K artifact), or when
        # sigma is a free parameter instead of a network -- see
        # reset_parameters / _current_scaling_factors. In the first case
        # _predict_scaling_factors and train_step will fail; that's expected.
        self.scaling_network = scaling_network.to(device) if scaling_network is not None else None
        self.brownian_bridge = brownian_bridge
        self.loss_fn = loss_fn
        self.device = device
        self.img_size = img_size
        self.lr = lr
        # weight_decay defaults to the 1e-5 the network optimizer below always
        # used. The non-amortized path overrides it to 0 -- see
        # reset_parameters, where the reason is measured out.
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.guidance_scale = guidance_scale
        # Mixed precision: fp16 activations for the network forward + warp + loss;
        # the SDE coefficient math (sigma_sq, delta_t) is forced to fp32 inside
        # brownian_bridge.run_posterior_reverse_process for numerical safety.
        self.use_amp = bool(use_amp) and (str(device) == 'cuda')

        # Posterior sampler used by compute_brownian_bridge_velocities. Setting
        # it here rather than per call site means every consumer (train_step,
        # compute_mean_velocity, the visualizers, main.py, test*.py) switches
        # together. See _langevin_velocities for what changes under 'langevin'.
        if sampler not in ('reverse_sde', 'langevin'):
            raise ValueError(f"Unknown sampler: {sampler!r}")
        self.sampler = sampler
        self.langevin_kw = dict(langevin_kw or {})
        self._langevin_notice_shown = False

        # Softplus maps the unconstrained free parameter to a positive sigma.
        # beta=0.5 matches the output head ScalingFactorNetwork uses
        # (networks.py:122), so the attainable sigma range is identical whether
        # sigma comes from the network or from s_raw.
        self.softplus = nn.Softplus(beta=0.5)

        # The non-amortized sigma. Allocated per minibatch by
        # reset_parameters() -- batch sizes are ragged on the final batch, so
        # the parameter cannot be sized at construction. Stays None whenever a
        # scaling network is in use.
        self.s_raw = None

        # Which optimizer reset_parameters() builds for the free sigma. Only
        # the non-amortized path reads this; the network optimizer below is
        # always Adam, unchanged.
        #
        # 'adam' is what the reference package uses. Note what Adam does to a
        # parameter that IS the output: its update lr*m/(sqrt(v)+eps) is
        # designed to EQUALIZE step sizes across parameters, so a pixel with
        # 10x the gradient does not get 10x the step. For network weights that
        # is the point; here each parameter is a voxel of sigma, so the
        # gradient-magnitude differences that ought to become sigma contrast
        # are normalized away. Measured per-element |grad| is 5.93e-09 against
        # Adam's eps=1e-8, i.e. right in the regime where that compression is
        # strongest, and the resulting map has IQR/median 0.33 against the
        # amortized 0.74.
        #
        # 'sgd' keeps step proportional to gradient, so relative magnitudes
        # survive into sigma. It needs a very different lr: Adam's step is
        # ~lr regardless of gradient scale, SGD's is lr*|g| with |g| ~ 6e-09,
        # so matching Adam's effective step needs lr larger by ~1/|g|.
        if sigma_optimizer not in ('adam', 'sgd'):
            raise ValueError(
                f"sigma_optimizer must be 'adam' or 'sgd', got "
                f"{sigma_optimizer!r}")
        self.sigma_optimizer = sigma_optimizer
        self.sigma_momentum = float(sigma_momentum)

        if scaling_network is not None:
            self.optimizer = torch.optim.Adam(
                self.scaling_network.parameters(),
                lr=lr,
                weight_decay=self.weight_decay
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6
            )
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        else:
            self.optimizer = None
            self.scheduler = None
            self.scaler = None

    def inverse_softplus(self, value):
        """
        The s_raw that makes softplus(s_raw) equal `value`.

        Same algebra as SigmaRegressionNetwork.set_output_bias (networks.py:249):
        softplus is (1/beta) log(1 + exp(beta*b)), which inverts to
        b = (1/beta) log(exp(beta*v) - 1).
        """
        beta = float(self.softplus.beta)
        v = max(float(value), 1e-6)
        return float(np.log(np.expm1(beta * v)) / beta)

    def reset_parameters(self, batch_size, init_sigma=None):
        """
        Allocate a fresh NON-AMORTIZED sigma parameter + optimizer for one
        minibatch. Ported from uncertainty_sd_combined_acdc_non_amortized.

        This replaces the amortized network entirely: each minibatch's K inner
        iterations start from a fresh init (test-time training, no transfer
        between minibatches). Because s_raw's leading dim is the batch size, it
        must be rebuilt per batch rather than reset.

        init_sigma sets WHERE sigma starts, via the inverse softplus, with the
        Xavier draw kept as jitter around it. Leaving it None reproduces the
        ported behaviour, s_raw ~ 0, i.e. sigma = softplus(0) = 1.386 -- and
        that is a poor starting point: the amortized network settles near
        s = 0.857, so a Xavier start overshoots by 62% and spends the
        iteration budget walking back down. This is the same waste
        SigmaRegressionNetwork.set_output_bias (networks.py:249) exists to
        avoid, whose docstring puts it as "starting at the target mean rather
        than at softplus(0) = 1.386 matters when the targets are small".
        main_langevin.py passes --sigma_init for this. Note the two
        candidate "principled" values disagree and the discrepancy is
        unresolved: the zero-residual stationary point of the loss this
        repository descends is lambda_scale/(alpha_scale + 1.5) -- 2.0 at the
        default lambda_scale 3 -- while mcem.py::sigma_init's closed form
        2*beta/(2*alpha + T + 1) gives 0.400 at the same hyperparameters, and
        it is 0.400 that is empirically demonstrated to land in the amortized
        network's 0.4-1.5 range. Hence a plain float rather than a formula.

        lr / weight_decay come from __init__. weight_decay should be 0 here,
        NOT the 1e-5 used for network weights: on a free sigma parameter Adam's
        decay term is wd*s_raw, which near init (|s_raw| ~ 0.02) is ~2e-7 per
        element -- an order of magnitude LARGER than the ~1.8e-8 per-element
        loss gradient. It therefore dominates the objective and pins s_raw at
        0, i.e. sigma at softplus(0) = 1.386 uniform, with no spatial
        structure. Measured: at lr=1e-4..1e-2 with wd=1e-5 the loss is flat and
        sigma's spread collapses (std 0.011 -> 0.0007); with wd=0 the loss
        decreases and sigma develops structure (std 0.011 -> 0.16 at lr=1e-2).
        This does not arise for the amortized U-Net, whose shared weights
        aggregate the per-pixel signal across all H*W pixels.
        """
        if self.scaling_network is not None:
            raise RuntimeError(
                "reset_parameters() allocates the non-amortized sigma and its "
                "optimizer, but this trainer holds a scaling network -- "
                "stepping both would clobber the network's optimizer. "
                "Construct the trainer with scaling_network=None to use the "
                "free parameter.")

        B, H = int(batch_size), self.img_size

        # Xavier uniform over the flattened spatial map, matching
        # uncertainty_sde_directOptim/trainer.py:50-51 (there [2, H*W]; here the
        # per-subject maps are stacked into the fan-in dim so every subject gets
        # the same init distribution).
        s_init = torch.empty(B, 2, H, H, device=self.device)
        torch.nn.init.xavier_uniform_(s_init.view(B * 2, H * H))
        # Shift the Xavier draw so softplus(s_init) starts at init_sigma. The
        # draw is kept as jitter rather than replaced, so the per-pixel
        # symmetry breaking the reference relies on survives; only the mean
        # moves. Xavier's spread here is ~0.016, so the jitter is small next
        # to the shift (e.g. -2.517 for init_sigma 0.5).
        if init_sigma is not None:
            s_init += self.inverse_softplus(init_sigma)
        self.s_raw = nn.Parameter(s_init)

        if self.sigma_optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                [self.s_raw],
                lr=self.lr,
                momentum=self.sigma_momentum,
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(
                [self.s_raw],
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def scheduler_step(self, val_loss):
        """Step the LR scheduler with the validation/epoch loss."""
        self.scheduler.step(val_loss)

    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def warp_image(self, source, velocity):
        """Warp source image with velocity field."""
        inshape = (self.img_size, self.img_size)
        return warp_image_with_velocity(source, velocity, inshape=inshape)

    def compute_brownian_bridge_velocities(self, v_reg_list, scaling_factors,
                                            source_img=None, target_imgs=None,
                                            num_samples=1):
        """
        Compute velocities at each cardiac frame using the Brownian Bridge formula
        with posterior reverse SDE (energy gradient correction).

        num_samples tells the bridge how many tiled sample blocks the batch axis
        holds, so antithetic noise pairing (if enabled) can line the pairs up
        with the `repeat(N, ...)` layout. It has no effect otherwise.
        """
        bb = self.brownian_bridge
        num_cardiac_frames = len(v_reg_list)

        if self.sampler == 'langevin':
            return self._langevin_velocities(v_reg_list, scaling_factors,
                                             source_img, target_imgs)

        v_0 = v_reg_list[0].clone()
        v_T = v_reg_list[-1].clone()

        # Run reverse process (posterior if images provided, prior otherwise)
        if source_img is not None and target_imgs is not None:
            trajectory = bb.run_posterior_reverse_process(
                v_reg_list, scaling_factors, v_0, v_T,
                source_img, target_imgs,
                gamma=self.gamma, guidance_scale=self.guidance_scale,
                device=self.device, num_samples=num_samples
            )
        else:
            trajectory = bb.run_reverse_process(
                v_reg_list, scaling_factors, v_0, v_T,
                device=self.device, num_samples=num_samples
            )

        # Extract v_t at cardiac frame boundaries from trajectory
        cardiac_frame_to_v_t = {}
        cardiac_frame_to_v_t[num_cardiac_frames - 1] = trajectory[0]  # v_T
        cardiac_frame_to_v_t[0] = trajectory[-1]  # v_0

        for traj_idx, t in enumerate(range(bb.T - 1, -1, -1)):
            if t > 0:
                cardiac_idx = bb.diffusion_timestep_to_cardiac_frame(t, num_cardiac_frames)
                frame_idx = int(round(cardiac_idx))
                if 0 <= frame_idx < num_cardiac_frames and frame_idx not in cardiac_frame_to_v_t:
                    cardiac_frame_to_v_t[frame_idx] = trajectory[traj_idx + 1]

        # Build output lists
        v_t_list = []
        std_tilde_list = self._marginal_std_from_scaling(scaling_factors, num_cardiac_frames)
        for frame_idx in range(num_cardiac_frames):
            if frame_idx in cardiac_frame_to_v_t:
                v_t_list.append(cardiac_frame_to_v_t[frame_idx])
            else:
                v_t_list.append(v_reg_list[frame_idx])

        return v_t_list, std_tilde_list

    def langevin_sigma(self, scaling_factors, keep_grad=False):
        """
        scaling_factors s -> the diffusion variance sigma the sampler takes.

        The reverse SDE consumes s and internally sets sigma_sq = 2s/T
        (brownian_bridge.py:414); the analytic marginal 2 s (m - m^2) is the same
        statement. `langevin_velocities` is parameterized directly in sigma with
        Var(eta_t) = sigma t(1 - t/T), so the matching conversion is sigma = 2s/T
        -- langevin_kw's convention 'paper_sigma'. Passing s straight through
        ('code_s') gives a T/2 = 7x wider prior at T = 14.

        scaling_factors is [B, T+1, C, H, W]; frame 0 is pinned to zero by
        _assemble_scaling_factors_bb, so the free frames are averaged.
        """
        convention = self.langevin_kw.get('convention', 'paper_sigma')
        scale = 1.0 if convention == 'code_s' else (self.brownian_bridge.T / 2.0)
        s_free = scaling_factors[:, 1:].mean(dim=1)
        # keep_grad leaves sigma attached so langevin_velocities(grad_steps=N)
        # can hand a pathwise dL/dsigma back to the network, matching what the
        # reverse SDE gives through its own sigma_sq coefficient.
        if not keep_grad:
            s_free = s_free.detach()
        return (s_free / scale).clamp_min(1e-8)

    def langevin_sampler_kw(self):
        """
        langevin_kw with this trainer's defaults filled in, and the keys that
        are the CALLER's business removed: n_steps, burn_in, thin, convention,
        init and the return flags. What is left can be splatted straight into
        `langevin_velocities`.
        """
        kw = {k: v for k, v in self.langevin_kw.items()
              if k not in ('convention', 'burn_in', 'thin', 'n_steps',
                           'init', 'init_eta', 'return_latent', 'return_stats')}
        kw.setdefault('gamma', self.gamma)
        kw.setdefault('guidance_scale', self.guidance_scale)
        kw.setdefault('reg_weight', getattr(self.brownian_bridge, 'reg_weight', 0.1))
        kw.setdefault('inshape', (self.img_size, self.img_size))
        return kw

    def _langevin_chain_key(self, v_reg_list, source_img):
        """
        Identity of the minibatch a persisted Langevin chain belongs to.

        Chain state may only be carried across calls that sample the SAME data.
        Inside train_step's inner-iteration loop the sampler is called K times
        on one minibatch -- the scaling factors change between iterations, the
        images do not -- and that is where continuing the chain pays off, since
        it is also exactly the MC-EM alternation (E-step samples at the current
        sigma, M-step updates sigma).

        Across minibatches the subjects change. Reusing the previous batch's
        equilibrium would seed each new subject's chain at a different
        subject's posterior, and with a thinning interval below the ~400-step
        relaxation time of the slowest bridge mode that bias does not wash out.
        Keying on a content fingerprint rather than on shapes alone makes the
        reset automatic for every entry point, without the call sites having to
        know the chain exists.
        """
        with torch.no_grad():
            fp = (float(source_img.float().sum()),
                  float(v_reg_list[-1].float().sum()))
        return (tuple(source_img.shape), len(v_reg_list), fp)

    def _langevin_velocities(self, v_reg_list, scaling_factors,
                             source_img=None, target_imgs=None):
        """
        Langevin (ULA) posterior sampler, drop-in for the Eq. 14 reverse SDE.

        UNITS. `scaling_factors` is s, which the reverse SDE consumes and
        internally converts with sigma_sq = 2s/T (brownian_bridge.py:414); the
        analytic marginal above, 2 s (m - m^2) = (2s/T) t (1 - t/T), is the same
        statement. `langevin_velocities` is parameterized directly in sigma with
        Var(eta_t) = sigma t(1 - t/T), so the matching conversion is
        sigma = 2s/T -- mcem's 'paper_sigma'. Passing s straight through would
        give Langevin a T/2 (= 7x at T=14) wider prior than the reverse SDE.
        `convention='code_s'` in langevin_kw opts into the other reading.

        GRADIENTS. By default the ULA loop detaches sigma and runs under
        no_grad, so the returned velocities carry NO gradient back to the
        scaling network -- unlike the reverse SDE path, which is differentiable
        through the
        trajectory. This is not a defect: it is the MC-EM factorization, where
        the E-step samples v at fixed sigma and the M-step then fits sigma to
        those samples. `loss_fn.compute_loss` still receives `scaling_factors`
        directly, and its bridge_norm / inv_gamma_prior terms remain
        differentiable in s given detached v, so training still has a gradient
        path -- it is just the M-step's, not a pathwise one.

        MEASURED CONSEQUENCE, and why main_langevin.py no longer does this:
        with sigma driven only by bridge_norm + inv_gamma_prior, its balance
        (beta + vtilde^2/rho)/(alpha + 1.5) pins sigma near beta/1.5. Observed
        over 6 epochs, sigma froze at ~1.90 from epoch 3 and never acquired
        image structure, where the reverse SDE settles at ~0.50 with clear
        anatomy. `langevin_velocities(grad_steps=N)` restores the missing force
        by keeping sigma attached through eps and sqrt(2 eps) on the last N
        updates -- the score stays detached, mirroring how the reverse SDE
        detaches grad_E but keeps sigma_sq live. main_langevin.py uses that;
        this path (reached from main.py) still detaches.

        NO PRIOR MODE. The reverse SDE falls back to run_reverse_process when
        images are absent (used by visualize_epoch). Langevin has no likelihood
        to drop, so that case is served by an exact bridge draw from the same
        prior, which is what the reverse process samples there anyway.
        """
        num_cardiac_frames = len(v_reg_list)
        T = num_cardiac_frames - 1
        convention = self.langevin_kw.get('convention', 'paper_sigma')
        burn_in = int(self.langevin_kw.get('burn_in', 0))
        n_steps = int(self.langevin_kw.get('n_steps', 200))
        sigma = self.langevin_sigma(scaling_factors)
        kw = self.langevin_sampler_kw()

        std_tilde_list = self._marginal_std_from_scaling(scaling_factors, num_cardiac_frames)

        if not self._langevin_notice_shown:
            self._langevin_notice_shown = True
            print(f"[trainer] sampler='langevin': convention={convention}, "
                  f"burn_in={burn_in}, n_steps={n_steps}, kw={kw}")
            print("[trainer]   note: velocities are detached — the scaling "
                  "network is trained through the loss's direct dependence on "
                  "scaling_factors (M-step), not pathwise through the sampler.")

        if source_img is None or target_imgs is None:
            # Prior-only draw (visualize_epoch): no likelihood to sample against.
            eta = _prior_draw(v_reg_list, sigma, T)
            v_t_list = [v_reg_list[t] + eta[:, t] for t in range(num_cardiac_frames)]
            return v_t_list, std_tilde_list

        if burn_in <= 0:
            # No chain kept: every call starts from a fresh prior draw and runs
            # n_steps, so successive calls give independent samples.
            v_t_list, _ = langevin_velocities(
                v_reg_list, sigma, source_img, target_imgs,
                n_steps=n_steps, init='prior', **kw)
            return v_t_list, std_tilde_list

        # One chain per minibatch, kept alive across calls:
        #   first call for a minibatch -> run burn_in steps, discard that state
        #   every call                 -> advance n_steps, return where it lands
        # Sample i of a minibatch therefore sits at burn_in + i*n_steps steps.
        #
        # main_langevin.py does NOT come through here: it reads
        # --langevin_steps as the total chain length and keeps every
        # post-burn-in state (see its `chain_samples`). This path exists for
        # main.py, whose compute_brownian_bridge_velocities contract is one
        # trajectory per call.
        key = self._langevin_chain_key(v_reg_list, source_img)
        prev = getattr(self, '_langevin_state', None)
        if prev is not None and prev[0] == key:
            xi = prev[1]                                     # same data: continue
        else:                                                # new data: burn in
            _, _, xi = langevin_velocities(
                v_reg_list, sigma, source_img, target_imgs,
                n_steps=burn_in, init='prior', return_latent=True, **kw)

        v_t_list, _, xi = langevin_velocities(
            v_reg_list, sigma, source_img, target_imgs,
            n_steps=n_steps, init_eta=xi, return_latent=True, **kw)
        self._langevin_state = (key, xi)
        return v_t_list, std_tilde_list

    def compute_mean_velocity(self, v_reg_list, scaling_factors_bb,
                              source_img, target_imgs, num_samples):
        """
        Vectorized N-sample mean velocity from the posterior reverse SDE.

        Stacks every input N times along the batch dim (effective batch =
        N*B), then makes ONE SDE call. Each reverse step's
        `torch.randn_like(v_t)` then draws N*B independent Gaussians, which
        is statistically identical to N i.i.d. trajectories per original
        sample. The output is reshaped back to [N, B, ...] and averaged
        along the N axis.

        Eliminates the Python for-loop over samples and lets the GPU
        saturate larger kernels — typically 2-5x faster than the loop
        variant. Total memory is unchanged (same total tensors retained
        for backward).

        Returns:
            v_bar_list:  list of [B, 2, H, W] (gradient-attached)
            u_list:      list of [B, 2, H, W] (detached, sample std per channel)
        """
        N = int(num_samples)
        if N <= 1:
            v_t_list, _ = self.compute_brownian_bridge_velocities(
                v_reg_list, scaling_factors_bb,
                source_img=source_img, target_imgs=target_imgs,
            )
            u_list = [torch.zeros_like(v) for v in v_t_list]
            return v_t_list, u_list

        B = v_reg_list[0].shape[0]

        # Repeat each input N times along the batch dim. `repeat` is
        # autograd-aware, so gradients from the [N*B, ...] graph fold back
        # into the original [B, ...] tensor by summing over the repeated
        # axis — exactly what we want for v_bar = mean over N.
        def _rep(t):
            return t.repeat(N, *([1] * (t.dim() - 1)))

        v_reg_NB           = [_rep(v) for v in v_reg_list]
        scaling_factors_NB = _rep(scaling_factors_bb)
        source_img_NB      = _rep(source_img)
        target_imgs_NB     = [_rep(t) for t in target_imgs]

        v_t_list_NB, _ = self.compute_brownian_bridge_velocities(
            v_reg_NB, scaling_factors_NB,
            source_img=source_img_NB, target_imgs=target_imgs_NB,
            num_samples=N,
        )

        v_bar_list = []
        u_list = []
        for v_NB in v_t_list_NB:
            # v_NB: [N*B, 2, H, W] -> [N, B, 2, H, W]
            v_view = v_NB.view(N, B, *v_NB.shape[1:])
            v_bar = v_view.mean(dim=0)                          # gradient-attached
            with torch.no_grad():
                centered = v_view.detach() - v_bar.detach().unsqueeze(0)
                u = (centered ** 2).mean(dim=0).clamp_min(0.0).sqrt()
            v_bar_list.append(v_bar)
            u_list.append(u)

        return v_bar_list, u_list

    def _marginal_std_from_scaling(self, scaling_factors_bb, num_frames):
        """
        Analytical bridge marginal std at each cardiac frame from the current
        scaling factors. Mirrors lines 115-125 of compute_brownian_bridge_velocities
        but avoids re-running the SDE.
        """
        out = []
        last_idx = scaling_factors_bb.shape[1] - 1
        for t in range(num_frames):
            m = t / (num_frames - 1) if num_frames > 1 else 0
            s = scaling_factors_bb[:, min(t, last_idx)]
            if t == 0 or t == num_frames - 1:
                out.append(torch.zeros_like(s))
            else:
                var = 2 * s * (m - m ** 2)
                out.append(torch.sqrt(torch.clamp(var, min=0.0)))
        return out

    def _predict_scaling_factors(self, source_img, target_imgs, v_reg_list, v_current_list=None):
        """
        Predict scaling factors from images, registered velocity, and (optional)
        current velocity v^k. If v_current_list is None, v_current = v_reg.
        """
        # Exclude t=0 (zero velocity) -- network sees only the T registration velocities
        v_reg_no_t0 = v_reg_list[1:]  # List of T tensors [B, 2, H, W]
        v_cur_no_t0 = v_reg_no_t0 if v_current_list is None else v_current_list[1:]

        scaling_factors_raw = self.scaling_network(
            source_img, target_imgs,
            v_reg_list=v_reg_no_t0,
            v_current_list=v_cur_no_t0,
        )
        # Clamp to prevent extreme values that cause NaN in reverse SDE
        scaling_factors_raw = torch.clamp(scaling_factors_raw, min=1e-4, max=1e4)

        # Expand to [B, T, 2, H, W] by repeating across time
        num_time_steps = len(v_reg_list) - 1  # exclude t=0
        scaling_factors = scaling_factors_raw.unsqueeze(1).expand(-1, num_time_steps, -1, -1, -1)
        zero_scaling = torch.zeros_like(scaling_factors[:, 0:1])
        scaling_factors_bb = torch.cat([zero_scaling, scaling_factors], dim=1)

        return scaling_factors_raw, scaling_factors, scaling_factors_bb

    def _current_scaling_factors(self, num_cardiac_frames):
        """
        Read the current sigma straight off the free parameter -- the
        non-amortized replacement for _predict_scaling_factors (which runs the
        U-Net on images + v_reg + v^k). Ported from
        uncertainty_sd_combined_acdc_non_amortized.

        sigma = clamp(Softplus(s_raw), 1e-4, 1e4). Softplus already guarantees
        positivity; the clamp is kept from the amortized path because sigma
        near zero underflows the 1/delta_t and log(delta_t) terms in the loss.

        The returned tensors stay attached to s_raw, so both the explicit sigma
        terms in the loss and the path-wise (reparameterized) sampler gradients
        flow back into the parameter.

        Same (raw, scaling_factors, scaling_factors_bb) triple
        _predict_scaling_factors returns, so the two are interchangeable at
        every call site.
        """
        if self.s_raw is None:
            raise RuntimeError(
                "_current_scaling_factors called before reset_parameters("
                "batch_size). The sigma parameter is allocated per minibatch.")
        scaling_factors_raw = self.softplus(self.s_raw)
        scaling_factors_raw = torch.clamp(scaling_factors_raw, min=1e-4, max=1e4)
        scaling_factors, scaling_factors_bb = self._expand_scaling_from_raw(
            scaling_factors_raw, num_cardiac_frames)
        return scaling_factors_raw, scaling_factors, scaling_factors_bb

    def _expand_scaling_from_raw(self, scaling_factors_raw, num_cardiac_frames):
        """
        Rebuild (scaling_factors, scaling_factors_bb) from a raw [B, 2, H, W]
        prediction. Same algebra as the tail of _predict_scaling_factors, split
        out so the chunked path can re-derive them from a detached leaf without
        re-running the network.
        """
        num_time_steps = num_cardiac_frames - 1
        scaling_factors = scaling_factors_raw.unsqueeze(1).expand(
            -1, num_time_steps, -1, -1, -1)
        zero_scaling = torch.zeros_like(scaling_factors[:, 0:1])
        scaling_factors_bb = torch.cat([zero_scaling, scaling_factors], dim=1)
        return scaling_factors, scaling_factors_bb

    def _delta_tilde_per_frame(self, scaling_factors_bb, num_cardiac_frames):
        """delta_tilde at each interior cardiac frame (see train_step)."""
        bb = self.brownian_bridge
        out = {}
        for frame_idx in range(1, num_cardiac_frames - 1):
            t_diffusion = int(round(frame_idx * bb.T / (num_cardiac_frames - 1)))
            s_t = bb.interpolate_scaling_factors(scaling_factors_bb, frame_idx)
            s_t_minus_1 = bb.interpolate_scaling_factors(scaling_factors_bb, frame_idx - 1)
            _, _, _, delta_tilde = bb.compute_coefficients(t_diffusion, s_t, s_t_minus_1)
            out[frame_idx] = delta_tilde
        return out

    def _chunked_iteration(self, source_img, target_imgs, v_reg_list,
                           v_current_list, num_cardiac_frames,
                           num_train_samples, sample_chunk):
        """
        One inner iteration with the N posterior samples split into chunks.

        Why this exists: the unchunked path tiles all N samples at once, so peak
        memory grows as N*B and B=128 with N=500 needs ~1.2 TB. Here the N
        samples are processed `sample_chunk` at a time; each chunk runs its own
        SDE + loss + backward and its graph is freed before the next chunk
        starts, so peak memory grows as sample_chunk*B instead.

        Exactness. All chunks share one network forward. The only tensor
        connecting them to the parameters is scaling_factors_raw, so each chunk
        backprops into a DETACHED fp32 copy of it (`sf_leaf`); the accumulated
        dL/d(sf_raw) is pushed through the network once at the end. Each chunk's
        loss is weighted C_j/N, which is exact for both term families in
        compute_loss: v-dependent terms are .mean() reductions over the tiled
        batch (weighted chunk means sum to the full-N mean), and sigma-only
        terms such as inv_gamma_prior are invariant under tiling, so the weights
        sum to 1 and reproduce the term unchanged. v_bar and u are accumulated
        as running sum / sum-of-squares over all N.

        Difference from the unchunked path: there the loss built at iteration
        k-1 is backpropagated at iteration k (one-step-stale gradients, see the
        `if k >= 1` block in train_step). Chunking cannot defer that — holding
        M chunk graphs across the loop boundary is exactly the memory this is
        meant to avoid — so here the loss is backpropagated in the iteration
        that produced it. Same gradient, applied one step earlier.

        Returns a dict with the per-iteration products train_step needs.
        """
        N = int(num_train_samples)
        C = max(1, int(sample_chunk))
        B = v_reg_list[0].shape[0]

        amp_ctx = (torch.cuda.amp.autocast(enabled=self.use_amp)
                   if self.use_amp else _NullContext())

        # Single network forward for the whole iteration.
        with amp_ctx:
            scaling_factors_raw, _, _ = self._predict_scaling_factors(
                source_img, target_imgs, v_reg_list,
                v_current_list=v_current_list,
            )

        # Detached fp32 leaf: every chunk's backward lands in sf_leaf.grad,
        # which autograd accumulates across chunks for free.
        sf_leaf = scaling_factors_raw.detach().float().requires_grad_(True)

        # Running moments of v_t over all N samples (fp32, no autograd).
        sum1 = [None] * num_cardiac_frames
        sum2 = [None] * num_cardiac_frames

        loss_accum = {}
        total_loss_val = 0.0

        start = 0
        while start < N:
            C_j = min(C, N - start)
            weight = C_j / N

            with amp_ctx:
                sf_j, sf_bb_j = self._expand_scaling_from_raw(
                    sf_leaf, num_cardiac_frames)
                delta_tilde_j = self._delta_tilde_per_frame(
                    sf_bb_j, num_cardiac_frames)

                def _rep(t):
                    return t.repeat(C_j, *([1] * (t.dim() - 1)))

                v_reg_C       = [_rep(v) for v in v_reg_list]
                source_img_C  = _rep(source_img)
                target_imgs_C = [_rep(t) for t in target_imgs]
                sf_bb_C       = _rep(sf_bb_j)
                sf_C          = _rep(sf_j)
                delta_tilde_C = {fi: _rep(d) for fi, d in delta_tilde_j.items()}

                v_t_list_C, _ = self.compute_brownian_bridge_velocities(
                    v_reg_C, sf_bb_C,
                    source_img=source_img_C, target_imgs=target_imgs_C,
                    num_samples=C_j,
                )

                warped_imgs_C = [self.warp_image(source_img_C, v_t)
                                 for v_t in v_t_list_C]
                loss_C, loss_dict_C = self.loss_fn.compute_loss(
                    source_img_C, target_imgs_C,
                    warped_imgs_C[1:], v_t_list_C[1:], sf_C,
                    delta_tilde_per_frame=delta_tilde_C,
                    v_t_list=v_t_list_C, v_reg_list=v_reg_C,
                    num_cardiac_frames=num_cardiac_frames,
                )

            # Backward outside autocast; frees this chunk's graph.
            self.scaler.scale(loss_C * weight).backward()

            with torch.no_grad():
                for i, v_C in enumerate(v_t_list_C):
                    v_view = v_C.detach().float().view(C_j, B, *v_C.shape[1:])
                    s1 = v_view.sum(dim=0)
                    s2 = (v_view * v_view).sum(dim=0)
                    if sum1[i] is None:
                        sum1[i], sum2[i] = s1, s2
                    else:
                        sum1[i] += s1
                        sum2[i] += s2

            total_loss_val += float(loss_C.detach()) * weight
            for key, val in loss_dict_C.items():
                loss_accum[key] = loss_accum.get(key, 0.0) + float(val) * weight

            del (v_t_list_C, warped_imgs_C, loss_C, v_reg_C, source_img_C,
                 target_imgs_C, sf_bb_C, sf_C, delta_tilde_C, sf_j, sf_bb_j,
                 delta_tilde_j)
            start += C_j

        # Push the accumulated dL/d(scaling_factors_raw) through the network.
        if sf_leaf.grad is not None:
            scaling_factors_raw.backward(
                gradient=sf_leaf.grad.to(scaling_factors_raw.dtype))

        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.scaling_network.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # v_bar / u over all N samples, from the running moments.
        with torch.no_grad():
            v_bar_list = []
            u_train_list = []
            for i in range(num_cardiac_frames):
                mean = sum1[i] / N
                var = (sum2[i] / N - mean * mean).clamp_min(0.0)
                v_bar_list.append(mean.to(v_reg_list[0].dtype))
                u_train_list.append(var.sqrt().to(v_reg_list[0].dtype))

            _, sf_bb_full = self._expand_scaling_from_raw(
                scaling_factors_raw.detach(), num_cardiac_frames)
            std_tilde_list = self._marginal_std_from_scaling(
                sf_bb_full, num_cardiac_frames)
            warped_imgs = [self.warp_image(source_img, v_t) for v_t in v_bar_list]

        loss_accum['total'] = total_loss_val
        return {
            'total_loss': total_loss_val,
            'loss_dict': loss_accum,
            'v_bar_list': v_bar_list,
            'u_train_list': u_train_list,
            'std_tilde_list': std_tilde_list,
            'warped_imgs': warped_imgs,
            'scaling_factors_bb': sf_bb_full,
        }

    def inference_refine(self, source_img, target_imgs, v_reg_list,
                         num_iterations=1, num_samples=1, sample_chunk=0,
                         progress=True):
        """
        Algorithm 1, Inference band (lines 10-13). theta is frozen: for
        r = 0..R predict sigma^r = f_theta(I, v_hat, v^{r-1}), draw N posterior
        reverse-SDE trajectories, and set v^r <- v_bar = (1/N) sum_n v^{(n)}.
        No loss, no backward, no optimizer step.

        The final round's per-frame sample std is returned as the scalar
        uncertainty map u_t = sqrt((1/N) sum_n ||v^{(n)}_t - v_bar_t||^2)
        (Algorithm 1 line 14).

        Not wrapped in torch.no_grad(): brownian_bridge.compute_energy_gradient
        calls torch.autograd.grad on the warp, which needs grad mode enabled.
        Only the network forward runs under no_grad — theta is frozen, so no
        parameter graph is built and nothing accumulates. See the same note at
        main.py's posterior sampling loop.

        sample_chunk > 0 draws the N trajectories `sample_chunk` at a time,
        accumulating running sum / sum-of-squares instead of holding all N, so
        peak memory grows as sample_chunk*B rather than N*B (same motivation as
        _chunked_iteration).

        Returns:
            (scaling_factors_bb, v_bar_list, u_list)
            scaling_factors_bb is sigma^R, detached, [B, T+1, 2, H, W].
            v_bar_list / u_list are per-cardiac-frame lists of [B, 2, H, W].
        """
        if self.scaling_network is None:
            raise RuntimeError(
                "inference_refine needs a scaling network; the trainer was "
                "built with scaling_network=None.")

        self.scaling_network.eval()
        num_cardiac_frames = len(v_reg_list)
        N = max(1, int(num_samples))
        B = v_reg_list[0].shape[0]
        chunk = int(sample_chunk) if int(sample_chunk) > 0 else N

        # v^{0} <- v_hat (Algorithm 1 line 2).
        v_current_list = [v.clone().detach() for v in v_reg_list]

        scaling_factors_bb = None
        v_bar_list = None
        u_list = None

        rounds = tqdm(range(num_iterations), desc="  refine iters",
                      leave=False, position=1, disable=not progress)
        amp_ctx = (torch.cuda.amp.autocast(enabled=self.use_amp)
                   if self.use_amp else _NullContext())

        for _ in rounds:
            # sigma^r <- f_theta(I, v_hat, v^{r-1}). theta is frozen, so the
            # forward needs no graph.
            with torch.no_grad(), amp_ctx:
                _, _, scaling_factors_bb = self._predict_scaling_factors(
                    source_img, target_imgs, v_reg_list,
                    v_current_list=v_current_list,
                )
            scaling_factors_bb = scaling_factors_bb.detach()

            # Draw N trajectories, chunked, accumulating sum and sum-of-squares
            # per cardiac frame so v_bar and u never need all N in memory.
            sums    = [None] * num_cardiac_frames
            sum_sqs = [None] * num_cardiac_frames
            drawn = 0
            while drawn < N:
                n_c = min(chunk, N - drawn)

                def _rep(t):
                    return t.repeat(n_c, *([1] * (t.dim() - 1)))

                with amp_ctx:
                    v_t_list_NB, _ = self.compute_brownian_bridge_velocities(
                        [_rep(v) for v in v_reg_list],
                        _rep(scaling_factors_bb),
                        source_img=_rep(source_img),
                        target_imgs=[_rep(t) for t in target_imgs],
                        num_samples=n_c,
                    )

                with torch.no_grad():
                    for fi, v_NB in enumerate(v_t_list_NB):
                        v_view = v_NB.view(n_c, B, *v_NB.shape[1:]).detach().float()
                        # Accumulate relative to v_reg (constant across the N
                        # samples, and close to every v^{(n)}), so the
                        # sum-of-squares stays small and E[x^2]-E[x]^2 below
                        # does not lose precision to cancellation.
                        c = v_view - v_reg_list[fi].detach().float().unsqueeze(0)
                        s  = c.sum(dim=0)
                        ss = (c ** 2).sum(dim=0)
                        sums[fi]    = s  if sums[fi]    is None else sums[fi]    + s
                        sum_sqs[fi] = ss if sum_sqs[fi] is None else sum_sqs[fi] + ss

                del v_t_list_NB
                drawn += n_c

            with torch.no_grad():
                # Undo the v_reg offset for the mean; the variance is
                # translation invariant, so u needs no correction.
                v_bar_list = [
                    v_reg_list[fi].detach().float() + s / N
                    for fi, s in enumerate(sums)
                ]
                u_list = [
                    (ss / N - (s / N) ** 2).clamp_min(0.0).sqrt()
                    for ss, s in zip(sum_sqs, sums)
                ]
            del sums, sum_sqs

            # v^{r} <- {v_bar_t} (Algorithm 1 line 13).
            v_current_list = [v.detach() for v in v_bar_list]

        rounds.close()

        # No gradient may have reached theta.
        stray = [n for n, p in self.scaling_network.named_parameters()
                 if p.grad is not None]
        if stray:
            raise RuntimeError(
                f"inference_refine leaked gradients onto {len(stray)} "
                f"parameter(s), first: {stray[0]}")

        return (scaling_factors_bb.detach(),
                [v.detach() for v in v_bar_list],
                [u.detach() for u in u_list])

    def train_step(self, source_img, target_imgs, v_reg_list, inner_iterations=400,
                   num_train_samples=1, sample_chunk=0,
                   vis_save_path=None, vis_subject_idx=0, vis_interval=1):
        """
        Iterative MLE training: for k in range(K), predict scaling using
        (images, v_reg, v_current^k), draw N posterior reverse-SDE trajectories,
        average them into v_bar, backprop loss(v_bar), optimizer.step, then
        v_current^{k+1} = v_bar.detach().

        With num_train_samples=1 the loop collapses to a single trajectory
        (matches v2 behavior exactly).

        sample_chunk > 0 splits the N samples into chunks of that size, keeping
        peak memory at sample_chunk*B instead of N*B — this is what makes large
        N (e.g. 500) feasible at full batch. See _chunked_iteration for the
        exactness argument and for the one behavioral difference (gradients are
        applied in the iteration that produced them, not one iteration later).
        sample_chunk=0 (default) keeps the original unchunked path untouched.

        If vis_save_path is provided, save training visualization (target/warped/
        velocities/scaling/std_tilde/diff) every vis_interval iterations into that
        directory as `training_vis_iter{k:04d}.png`. The final iteration is always
        rendered; vis_interval=0 disables the periodic renders so that final one
        is the ONLY figure this call produces (each costs ~6 s of matplotlib, so
        this is what you want when the caller only needs one figure per epoch).

        Returns:
            (last_loss_dict, scaling_factors_bb)
            where scaling_factors_bb is s^K (detached) from the final iteration.
        """
        self.scaling_network.train()
        bb = self.brownian_bridge
        num_cardiac_frames = len(v_reg_list)

        # Initialize v_current = v_reg (detached, no grad).
        v_current_list = [v.clone().detach() for v in v_reg_list]

        last_loss_dict = None
        scaling_factors_bb = None

        inner_pbar = tqdm(range(inner_iterations), desc="  inner iters",
                          leave=False, position=1)
        amp_ctx = (torch.cuda.amp.autocast(enabled=self.use_amp)
                   if self.use_amp else _NullContext())

        use_chunked = int(sample_chunk) > 0 and int(num_train_samples) > 1

        for k in inner_pbar:
            self.optimizer.zero_grad()

            if use_chunked:
                out = self._chunked_iteration(
                    source_img, target_imgs, v_reg_list, v_current_list,
                    num_cardiac_frames, num_train_samples, sample_chunk,
                )
                loss_dict           = out['loss_dict']
                v_bar_list          = out['v_bar_list']
                u_train_list        = out['u_train_list']
                std_tilde_list      = out['std_tilde_list']
                warped_imgs         = out['warped_imgs']
                scaling_factors_bb  = out['scaling_factors_bb']
                v_t_list            = v_bar_list

                v_current_list = [v.detach() for v in v_t_list]

                loss_dict['mean_u'] = float(
                    torch.stack([u.mean() for u in u_train_list]).mean().item()
                )
                last_loss_dict = loss_dict
                inner_pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'sim': f"{loss_dict['similarity']:.4f}",
                    'bn': f"{loss_dict.get('bridge_norm', 0):.4f}",
                    'inv_gamma': f"{loss_dict['inv_gamma_prior']:.4f}",
                    'lr': f"{self.get_lr():.2e}",
                })

                if vis_save_path is not None and (
                        (vis_interval > 0 and k % vis_interval == 0)
                        or k == inner_iterations - 1):
                    self.visualize_training_step(
                        scaling_factors_bb, std_tilde_list, v_reg_list, v_t_list,
                        source_img, target_imgs, warped_imgs,
                        vis_save_path, epoch=k, subject_idx=vis_subject_idx,
                    )
                continue

            with amp_ctx:
                # Predict scaling using images, v_reg, and current v^k
                _, scaling_factors, scaling_factors_bb = self._predict_scaling_factors(
                    source_img, target_imgs, v_reg_list,
                    v_current_list=v_current_list,
                )

                # delta_tilde at each cardiac frame
                delta_tilde_per_frame = {}
                for frame_idx in range(1, num_cardiac_frames - 1):
                    t_diffusion = int(round(frame_idx * bb.T / (num_cardiac_frames - 1)))
                    s_t = bb.interpolate_scaling_factors(scaling_factors_bb, frame_idx)
                    s_t_minus_1 = bb.interpolate_scaling_factors(scaling_factors_bb, frame_idx - 1)
                    _, _, _, delta_tilde = bb.compute_coefficients(t_diffusion, s_t, s_t_minus_1)
                    delta_tilde_per_frame[frame_idx] = delta_tilde

                if k >= 1:
                    warped_imgs_NB = [self.warp_image(source_img_NB, v_t) for v_t in v_t_list_NB]
                    total_loss, loss_dict = self.loss_fn.compute_loss(
                        source_img_NB, target_imgs_NB,
                        warped_imgs_NB[1:], v_t_list_NB[1:], sf_NB,
                        delta_tilde_per_frame=delta_tilde_NB,
                        v_t_list=v_t_list_NB, v_reg_list=v_reg_NB,
                        num_cardiac_frames=num_cardiac_frames,
                    )
                    
                #if self.use_amp:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.scaling_network.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                # else:
                #     total_loss.backward()
                #     torch.nn.utils.clip_grad_norm_(self.scaling_network.parameters(), max_norm=1.0)
                #     self.optimizer.step()
                
                # Sample N posterior reverse-SDE trajectories and compute the
                # MEAN-OF-LOSS estimator of E_{v ~ p_σ}[L_one(v, σ)] (paper
                # Eq. 10, Algorithm 1 line 5). Implementation: tile every
                # input N times along the batch dim so the SDE produces N*B
                # independent samples in one call, then pass the tiled σ /
                # tiled v to compute_loss. The .mean() reductions inside
                # compute_loss naturally average v-dependent terms over the
                # N samples, while σ-only terms (inv_gamma_prior) are
                # invariant under tiling — together this is the unbiased
                # MC estimator of the expected loss. Preserves the Jensen
                # gap that drives σ to anatomical structure (the loss-of-
                # mean variant collapses σ to a uniform shape as N grows).
                #
                # N=1 keeps the simple non-tiled path (no extra repeat).
                if num_train_samples <= 1:
                    v_t_list, std_tilde_list = self.compute_brownian_bridge_velocities(
                        v_reg_list, scaling_factors_bb,
                        source_img=source_img, target_imgs=target_imgs,
                    )
                    u_train_list = None

                    # The loss/backward at the top of the NEXT iteration (the
                    # `if k >= 1` block) reads the _NB names, which only the
                    # N > 1 branch below binds. With N = 1 the tiling is the
                    # identity, so alias them here — without this, iteration
                    # k = 1 raises UnboundLocalError on v_t_list_NB.
                    v_t_list_NB     = v_t_list
                    source_img_NB   = source_img
                    target_imgs_NB  = target_imgs
                    sf_NB           = scaling_factors
                    delta_tilde_NB  = delta_tilde_per_frame
                    v_reg_NB        = v_reg_list

                    warped_imgs = [self.warp_image(source_img, v_t) for v_t in v_t_list]
                    total_loss, loss_dict = self.loss_fn.compute_loss(
                        source_img, target_imgs, warped_imgs[1:], v_t_list[1:], scaling_factors,
                        delta_tilde_per_frame=delta_tilde_per_frame,
                        v_t_list=v_t_list, v_reg_list=v_reg_list,
                        num_cardiac_frames=num_cardiac_frames,
                    )
                    v_bar_list = v_t_list
                else:
                    N = int(num_train_samples)
                    B = v_reg_list[0].shape[0]

                    def _rep(t):
                        return t.repeat(N, *([1] * (t.dim() - 1)))

                    v_reg_NB        = [_rep(v) for v in v_reg_list]
                    source_img_NB   = _rep(source_img)
                    target_imgs_NB  = [_rep(t) for t in target_imgs]
                    sf_bb_NB        = _rep(scaling_factors_bb)
                    sf_NB           = _rep(scaling_factors)
                    delta_tilde_NB  = {fi: _rep(d) for fi, d in delta_tilde_per_frame.items()}

                    # SDE at effective batch N*B — N i.i.d. trajectories per
                    # original batch element. σ stays in the graph via the
                    # tiled sf_bb_NB, so reparameterization gives path-wise
                    # gradients on σ for every sample.
                    v_t_list_NB, _ = self.compute_brownian_bridge_velocities(
                        v_reg_NB, sf_bb_NB,
                        source_img=source_img_NB, target_imgs=target_imgs_NB,
                        num_samples=N,
                    )

                    # warped_imgs_NB = [self.warp_image(source_img_NB, v_t) for v_t in v_t_list_NB]
                    # total_loss, loss_dict = self.loss_fn.compute_loss(
                    #     source_img_NB, target_imgs_NB,
                    #     warped_imgs_NB[1:], v_t_list_NB[1:], sf_NB,
                    #     delta_tilde_per_frame=delta_tilde_NB,
                    #     v_t_list=v_t_list_NB, v_reg_list=v_reg_NB,
                    #     num_cardiac_frames=num_cardiac_frames,
                    # )

                    # Reshape [N*B, ...] -> [N, B, ...] for v_bar / u (used
                    # for v_current^{r+1} and the uncertainty map). Detached
                    # so they don't appear in backward.
                    v_bar_list = []
                    u_train_list = []
                    with torch.no_grad():
                        for v_NB in v_t_list_NB:
                            v_view = v_NB.view(N, B, *v_NB.shape[1:]).detach()
                            v_bar = v_view.mean(dim=0)
                            centered = v_view - v_bar.unsqueeze(0)
                            u = (centered ** 2).mean(dim=0).clamp_min(0.0).sqrt()
                            v_bar_list.append(v_bar)
                            u_train_list.append(u)

                    std_tilde_list = self._marginal_std_from_scaling(
                        scaling_factors_bb, num_cardiac_frames
                    )

                    # Visualize the per-batch v_bar / warps, not the tiled
                    # versions — keeps figures at original batch size.
                    with torch.no_grad():
                        warped_imgs = [self.warp_image(source_img, v_t) for v_t in v_bar_list]
                    v_t_list = v_bar_list

            # if self.use_amp:
            #     self.scaler.scale(total_loss).backward()
            #     self.scaler.unscale_(self.optimizer)
            #     torch.nn.utils.clip_grad_norm_(self.scaling_network.parameters(), max_norm=1.0)
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
            # else:
            #     total_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(self.scaling_network.parameters(), max_norm=1.0)
            #     self.optimizer.step()

            # v_current^{k+1} = v_perturbed.detach() for next iteration
            # (When num_train_samples > 1, v_t_list holds the per-frame mean
            # v_bar; matches Algorithm 1 line 7: v^{r+1} <- {v_bar_t}.)
            v_current_list = [v.detach() for v in v_t_list]

            if k >=1:
                loss_dict['total'] = total_loss.item()
                if u_train_list is not None:
                    loss_dict['mean_u'] = float(
                        torch.stack([u.mean() for u in u_train_list]).mean().item()
                    )
                last_loss_dict = loss_dict

                inner_pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'sim': f"{loss_dict['similarity']:.4f}",
                    'bn': f"{loss_dict.get('bridge_norm', 0):.4f}",
                    'inv_gamma': f"{loss_dict['inv_gamma_prior']:.4f}",
                    'lr': f"{self.get_lr():.2e}",
                })

            # Per-iteration training visualization (optional)
            if vis_save_path is not None and (
                    (vis_interval > 0 and k % vis_interval == 0)
                    or k == inner_iterations - 1):
                self.visualize_training_step(
                    scaling_factors_bb, std_tilde_list, v_reg_list, v_t_list,
                    source_img, target_imgs, warped_imgs,
                    vis_save_path, epoch=k, subject_idx=vis_subject_idx,
                )

        inner_pbar.close()

        # The σ inside the loop is f_{θ_{k}}(v^{k}) — predicted at the START
        # of iteration k, before that iteration's optimizer.step(). So at
        # exit, scaling_factors_bb is σ^{K-1} (one step stale). Re-predict
        # once with the FINAL θ_K and the final v_current = v^K so the s^K
        # artifact reflects the fully-trained network.
        self.scaling_network.eval()
        with torch.no_grad():
            _, _, scaling_factors_bb = self._predict_scaling_factors(
                source_img, target_imgs, v_reg_list,
                v_current_list=v_current_list,
            )
        self.scaling_network.train()

        return last_loss_dict, scaling_factors_bb.detach()

    def visualize_training_step(self, scaling_factors, std_tilde_list, v_reg_list, v_t_list,
                                 source_img, target_imgs, warped_imgs,
                                 save_path, epoch, subject_idx=None):
        """Visualize scaling factors, std_tilde, and velocities during training."""
        os.makedirs(save_path, exist_ok=True)

        num_frames = len(v_reg_list)
        fig, axes = plt.subplots(7, num_frames, figsize=(2.5 * num_frames, 16))

        with torch.no_grad():
            batch_size = source_img.shape[0]
            if subject_idx is None:
                subject_idx = np.random.randint(0, batch_size)

            src = source_img[subject_idx, 0].cpu().numpy()

            std_mag_max = 0.0
            for t in range(num_frames):
                std = std_tilde_list[t][subject_idx].cpu().numpy()
                std_mag = np.sqrt(std[0] + std[1])
                std_mag_max = max(std_mag_max, std_mag.max())
            std_mag_max = max(std_mag_max, 1e-6)

            for t in range(num_frames):
                if t == 0:
                    target = src
                else:
                    target = target_imgs[t-1][subject_idx, 0].cpu().numpy()

                axes[0, t].imshow(target, cmap='gray', vmin=0, vmax=1)
                axes[0, t].set_title(f't={t}', fontsize=10)
                axes[0, t].axis('off')

                warped = warped_imgs[t][subject_idx, 0].cpu().numpy()
                axes[1, t].imshow(warped, cmap='gray', vmin=0, vmax=1)
                axes[1, t].axis('off')

                v_reg = v_reg_list[t][subject_idx].cpu().numpy()
                v_reg_mag = np.sqrt(v_reg[0]**2 + v_reg[1]**2)
                im2 = axes[2, t].imshow(v_reg_mag, cmap='jet')
                axes[2, t].axis('off')
                if t == num_frames - 1:
                    plt.colorbar(im2, ax=axes[2, t], fraction=0.046)

                s = scaling_factors[subject_idx, t].cpu().numpy()
                s_mag = np.sqrt(s[0]**2 + s[1]**2)
                axes[3, t].imshow(src, cmap='gray', vmin=0, vmax=1)
                im3 = axes[3, t].imshow(s_mag, cmap='viridis', alpha=0.7)
                axes[3, t].axis('off')
                if t == num_frames - 1:
                    plt.colorbar(im3, ax=axes[3, t], fraction=0.046)

                std = std_tilde_list[t][subject_idx].cpu().numpy()
                std_mag = np.sqrt(std[0] + std[1])
                im4 = axes[4, t].imshow(std_mag, cmap='hot', vmin=0.0, vmax=std_mag_max)
                axes[4, t].axis('off')
                if t == num_frames - 1:
                    plt.colorbar(im4, ax=axes[4, t], fraction=0.046)

                v_t = v_t_list[t][subject_idx].cpu().numpy()
                v_t_mag = np.sqrt(v_t[0]**2 + v_t[1]**2)
                im5 = axes[5, t].imshow(v_t_mag, cmap='jet')
                axes[5, t].axis('off')
                if t == num_frames - 1:
                    plt.colorbar(im5, ax=axes[5, t], fraction=0.046)

                diff = np.abs(warped - target)
                im6 = axes[6, t].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
                axes[6, t].axis('off')
                if t == num_frames - 1:
                    plt.colorbar(im6, ax=axes[6, t], fraction=0.046)

        axes[0, 0].set_ylabel('Target', fontsize=12)
        axes[1, 0].set_ylabel('Warped', fontsize=12)
        axes[2, 0].set_ylabel('|v_reg|', fontsize=12)
        axes[3, 0].set_ylabel('|s| (scale)', fontsize=12)
        axes[4, 0].set_ylabel('|std_tilde|', fontsize=12)
        axes[5, 0].set_ylabel('|v_t| (BB)', fontsize=12)
        axes[6, 0].set_ylabel('|warp-target|', fontsize=12)

        plt.suptitle(f'Epoch {epoch + 1} (Subject {subject_idx})', fontsize=14)
        plt.tight_layout()

        save_file = f"{save_path}/training_vis_iter{epoch:04d}.png"
        plt.savefig(save_file, dpi=100, bbox_inches='tight')
        plt.close()

    def visualize_epoch(self, source_img, target_imgs, v_reg_list, save_path, epoch, subject_idx=None):
        """Visualize at the end of an epoch."""
        self.scaling_network.eval()

        with torch.no_grad():
            _, _, scaling_factors_bb = self._predict_scaling_factors(
                source_img, target_imgs, v_reg_list
            )

            # Use prior reverse process for visualization (no energy gradient)
            v_t_list, std_tilde_list = self.compute_brownian_bridge_velocities(
                v_reg_list, scaling_factors_bb
            )

            warped_imgs = []
            for v_t in v_t_list:
                warped = self.warp_image(source_img, v_t)
                warped_imgs.append(warped)

            self.visualize_training_step(
                scaling_factors_bb, std_tilde_list, v_reg_list, v_t_list,
                source_img, target_imgs, warped_imgs,
                save_path, epoch, subject_idx
            )

        self.scaling_network.train()
