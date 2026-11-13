"""
Loss functions for learned scaling factor training with EM algorithm.

Contains:
- Grad: N-D gradient loss for velocity field regularization
- NCC: Local normalized cross correlation loss
- LTMALoss: Original combined loss function
- EMLoss: EM M-step loss based on the log joint posterior

The EM M-step loss (from BridgeUQ) is:
    L(theta) = sum_t [1/gamma * E(v_t(s_theta)) + |Omega|/2 * log(delta_tilde_t(theta))]
               + [beta / s_theta + (alpha+1) * log(s_theta)]

Components:
1. Registration energy with temperature: 1/gamma * E(v_t)
2. Bridge normalization: |Omega|/2 * log(delta_tilde_t)
3. Inverse gamma prior: beta/s + (alpha+1)*log(s)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def normalize_velocity(v):
    """Normalize velocity field to [0, 1] per-sample per-channel."""
    v_min = v.amin(dim=(-2, -1), keepdim=True)
    v_max = v.amax(dim=(-2, -1), keepdim=True)
    return (v - v_min) / (v_max - v_min + 1e-8)


def compute_image_gradient(image):
    """
    Compute normalized image gradient magnitude.

    Args:
        image: [B, 1, H, W] input image

    Returns:
        [B, 1, H, W] gradient magnitude normalized to [0, 1]
    """
    dy = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    dx = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])

    dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
    dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')

    grad_mag = torch.sqrt(dx**2 + dy**2 + 1e-8)

    B = grad_mag.shape[0]
    grad_mag_flat = grad_mag.view(B, -1)
    grad_min = grad_mag_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    grad_max = grad_mag_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    grad_mag = (grad_mag - grad_min) / (grad_max - grad_min + 1e-8)

    return grad_mag


class Grad:
    """N-D gradient loss for velocity field regularization."""

    def __init__(self, penalty='l2'):
        self.penalty = penalty

    def loss(self, y_pred):
        if len(y_pred.shape) == 4:
            dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
            dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx

            return (torch.mean(dx) + torch.mean(dy)) / 2.0
        else:
            raise ValueError(f"Expected 4D tensor, got shape {y_pred.shape}")


class NCC:
    """Local normalized cross correlation loss."""

    def __init__(self, win=9):
        self.win = [win, win]

    def loss(self, y_true, y_pred):
        Ii = y_true
        Ji = y_pred

        win = self.win
        sum_filt = torch.ones([1, 1, *win]).to(y_true.device)
        pad_no = win[0] // 2
        stride = (1, 1)
        padding = (pad_no, pad_no)

        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = F.conv2d(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = F.conv2d(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = F.conv2d(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = F.conv2d(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = F.conv2d(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return - torch.mean(cc)


class GlobalNCC:
    """Global (whole-image) normalized cross correlation loss."""

    def loss(self, y_true, y_pred):
        B = y_true.shape[0]
        I = y_true.reshape(B, -1)
        J = y_pred.reshape(B, -1)

        I = I - I.mean(dim=1, keepdim=True)
        J = J - J.mean(dim=1, keepdim=True)

        num = (I * J).sum(dim=1)
        den = torch.sqrt((I * I).sum(dim=1) * (J * J).sum(dim=1) + 1e-5)
        cc = num / den  # in [-1, 1]

        return 1.0 - cc.mean()


class SSIM:
    """Structural similarity index loss (2D, single-channel).

    Returns 1 - mean(SSIM) so the loss is in [0, 2] (typically near 0).
    Uses a Gaussian window following Wang et al., 2004.
    """

    def __init__(self, win=11, sigma=1.5, data_range=1.0):
        self.win = win
        self.sigma = sigma
        self.data_range = data_range
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

        coords = torch.arange(win, dtype=torch.float32) - (win - 1) / 2.0
        g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()
        g2d = g1d.unsqueeze(0) * g1d.unsqueeze(1)
        self._kernel = g2d.view(1, 1, win, win)

    def _ssim_cs(self, y_true, y_pred):
        """Returns (mean SSIM, mean contrast-structure term)."""
        kernel = self._kernel.to(y_true.device, y_true.dtype)
        pad = self.win // 2

        mu_x = F.conv2d(y_true, kernel, padding=pad)
        mu_y = F.conv2d(y_pred, kernel, padding=pad)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(y_true * y_true, kernel, padding=pad) - mu_x2
        sigma_y2 = F.conv2d(y_pred * y_pred, kernel, padding=pad) - mu_y2
        sigma_xy = F.conv2d(y_true * y_pred, kernel, padding=pad) - mu_xy

        l_map = (2 * mu_xy + self.C1) / (mu_x2 + mu_y2 + self.C1)
        cs_map = (2 * sigma_xy + self.C2) / (sigma_x2 + sigma_y2 + self.C2)
        ssim_map = l_map * cs_map
        return ssim_map.mean(), cs_map.mean()

    def loss(self, y_true, y_pred):
        ssim, _ = self._ssim_cs(y_true, y_pred)
        return 1.0 - ssim


class MS_SSIM:
    """Multi-scale SSIM loss (Wang et al., 2003).

    Combines SSIM computed at multiple scales via iterative avg-pool downsampling.
    Auto-reduces number of levels if the image is too small for the SSIM window.
    Returns 1 - MS-SSIM.
    """

    def __init__(self, win=11, sigma=1.5, data_range=1.0, weights=None):
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.ssim = SSIM(win=win, sigma=sigma, data_range=data_range)

    def loss(self, y_true, y_pred):
        weights = self.weights.to(y_true.device, y_true.dtype)
        # Auto-reduce levels so the coarsest image still fits the SSIM window
        min_dim = min(y_true.shape[-2], y_true.shape[-1])
        max_levels = max(1, int(np.floor(np.log2(min_dim / self.ssim.win))) + 1)
        n_levels = min(len(weights), max_levels)
        w = weights[:n_levels]
        w = w / w.sum()

        mssim = torch.tensor(1.0, device=y_true.device, dtype=y_true.dtype)
        x, y = y_true, y_pred
        for i in range(n_levels):
            ssim_i, cs_i = self.ssim._ssim_cs(x, y)
            if i < n_levels - 1:
                # Use CS term at intermediate scales, then downsample
                mssim = mssim * torch.clamp(cs_i, min=1e-8) ** w[i]
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                y = F.avg_pool2d(y, kernel_size=2, stride=2)
            else:
                # Full SSIM at coarsest scale
                mssim = mssim * torch.clamp(ssim_i, min=1e-8) ** w[i]
        return 1.0 - mssim


class MI:
    """Differentiable Mutual Information loss via Parzen-window soft histograms.

    Assumes inputs are roughly in [0, 1]. Computes a soft joint histogram using
    Gaussian kernels, then MI = H(I) + H(J) - H(I, J). When normalized=True,
    returns 1 - NMI with NMI in [0, 1]; otherwise returns -MI.
    """

    def __init__(self, num_bins=32, sigma=None, normalized=True):
        self.num_bins = num_bins
        self.sigma = sigma if sigma is not None else 1.0 / num_bins
        self.normalized = normalized

    def loss(self, y_true, y_pred):
        B = y_true.shape[0]
        I = y_true.reshape(B, -1)
        J = y_pred.reshape(B, -1)
        N = I.shape[1]

        bins = torch.linspace(0.0, 1.0, self.num_bins, device=I.device, dtype=I.dtype)
        wI = torch.exp(-((I.unsqueeze(-1) - bins) / self.sigma) ** 2)
        wJ = torch.exp(-((J.unsqueeze(-1) - bins) / self.sigma) ** 2)
        wI = wI / (wI.sum(dim=-1, keepdim=True) + 1e-10)
        wJ = wJ / (wJ.sum(dim=-1, keepdim=True) + 1e-10)

        p_joint = torch.einsum('bni,bnj->bij', wI, wJ) / float(N)
        p_joint = p_joint + 1e-10
        p_I = p_joint.sum(dim=2)
        p_J = p_joint.sum(dim=1)

        H_I = -(p_I * torch.log(p_I)).sum(dim=1)
        H_J = -(p_J * torch.log(p_J)).sum(dim=1)
        H_IJ = -(p_joint * torch.log(p_joint)).sum(dim=(1, 2))
        mi = H_I + H_J - H_IJ

        if self.normalized:
            nmi = 2.0 * mi / (H_I + H_J + 1e-10)
            return 1.0 - nmi.mean()
        return -mi.mean()


class NMI:
    """Normalized Mutual Information (Studholme et al., 1999).

    NMI_studholme(I, J) = (H(I) + H(J)) / H(I, J), in [1, 2] for well-registered
    images (2 = identical, 1 = independent). Returns loss = 2 - NMI so that the
    loss is in [0, 1] with 0 at a perfect match.

    Computed via the same Parzen-window soft histogram as MI.
    """

    def __init__(self, num_bins=32, sigma=None):
        self.num_bins = num_bins
        self.sigma = sigma if sigma is not None else 1.0 / num_bins

    def loss(self, y_true, y_pred):
        B = y_true.shape[0]
        I = y_true.reshape(B, -1)
        J = y_pred.reshape(B, -1)
        N = I.shape[1]

        bins = torch.linspace(0.0, 1.0, self.num_bins, device=I.device, dtype=I.dtype)
        wI = torch.exp(-((I.unsqueeze(-1) - bins) / self.sigma) ** 2)
        wJ = torch.exp(-((J.unsqueeze(-1) - bins) / self.sigma) ** 2)
        wI = wI / (wI.sum(dim=-1, keepdim=True) + 1e-10)
        wJ = wJ / (wJ.sum(dim=-1, keepdim=True) + 1e-10)

        p_joint = torch.einsum('bni,bnj->bij', wI, wJ) / float(N)
        p_joint = p_joint + 1e-10
        p_I = p_joint.sum(dim=2)
        p_J = p_joint.sum(dim=1)

        H_I = -(p_I * torch.log(p_I)).sum(dim=1)
        H_J = -(p_J * torch.log(p_J)).sum(dim=1)
        H_IJ = -(p_joint * torch.log(p_joint)).sum(dim=(1, 2))

        nmi = (H_I + H_J) / (H_IJ + 1e-10)  # in [1, 2]
        return 2.0 - nmi.mean()


class DeepSim:
    """DeepSim semantic similarity loss (Czolbe et al., MIDL 2021).

    Computes similarity in the feature space of a frozen pretrained network
    rather than at the pixel level. The paper trains a U-Net-style
    autoencoder on segmentation labels from the same domain. As a drop-in
    default when no custom extractor is provided, this implementation uses
    torchvision VGG16 (ImageNet pretrained) features at a few early layers —
    this is the common "perceptual-loss" proxy reported alongside DeepSim in
    several follow-up works.

    Args:
        feature_extractor: Optional nn.Module or callable mapping
            [B, 1, H, W] or [B, 3, H, W] -> a feature tensor [B, C', h, w] or a
            list/tuple of such tensors. Should be frozen by the caller.
        similarity: 'ncc' (zero-mean cosine) or 'cosine' — used inside the
            feature space. Default 'ncc'.
        vgg_layers: VGG16 feature-layer indices to tap when no custom
            extractor is provided. Defaults to (8, 15) ≈ relu2_2 + relu3_3.
    """

    def __init__(self, feature_extractor=None, similarity='ncc', vgg_layers=(8, 15)):
        if similarity not in ('ncc', 'cosine', 'mse', 'l1'):
            raise ValueError(f"DeepSim similarity must be 'ncc', 'cosine', 'mse', or 'l1'; got {similarity}")
        self.similarity = similarity
        self._owns_extractor = feature_extractor is None
        self._vgg_loaded = False
        self._vgg_layers = tuple(vgg_layers)
        self.feature_extractor = feature_extractor  # may be None until first loss call

        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _load_vgg(self, device):
        import torchvision.models as tvm
        try:
            vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT).features
        except Exception:
            vgg = tvm.vgg16(pretrained=True).features  # older torchvision
        for p in vgg.parameters():
            p.requires_grad_(False)
        vgg.eval().to(device)
        self.feature_extractor = vgg
        self._vgg_loaded = True

    def _prepare(self, img):
        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)
        mean = self._mean.to(img.device, img.dtype)
        std = self._std.to(img.device, img.dtype)
        return (img - mean) / std

    def _vgg_features(self, img):
        feats = []
        x = img
        max_layer = max(self._vgg_layers)
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i in self._vgg_layers:
                feats.append(x)
            if i >= max_layer:
                break
        return feats

    def _extract(self, img):
        if self._owns_extractor:
            if not self._vgg_loaded:
                self._load_vgg(img.device)
            return self._vgg_features(self._prepare(img))
        out = self.feature_extractor(img)
        return out if isinstance(out, (list, tuple)) else [out]

    @staticmethod
    def _channel_l2_normalize(f, eps=1e-8):
        """Unit-normalize feature vectors along the channel dimension per spatial location.

        Input: [B, C, H, W] -> each [B, :, h, w] vector has L2 norm 1. Makes the
        subsequent MSE/L1 distance scale-invariant to per-layer feature magnitude
        (this is the LPIPS-style convention).
        """
        norm = torch.sqrt((f * f).sum(dim=1, keepdim=True) + eps)
        return f / norm

    def _per_layer_loss(self, fI, fJ):
        """Return a per-layer scalar loss already oriented for minimization.

        For similarity-style aggregations (ncc, cosine) returns 1 - sim.
        For distance-style aggregations (mse, l1) returns the distance between
        channel-L2-normalized feature maps so that the scale is comparable to
        pixel-level MSE/L1.
        """
        if self.similarity in ('mse', 'l1'):
            a = self._channel_l2_normalize(fI)
            b = self._channel_l2_normalize(fJ)
            diff = a - b
            if self.similarity == 'mse':
                return (diff * diff).mean()
            return diff.abs().mean()

        # ncc / cosine: normalized inner product per channel, averaged
        B, C = fI.shape[:2]
        a = fI.reshape(B, C, -1)
        b = fJ.reshape(B, C, -1)
        if self.similarity == 'ncc':
            a = a - a.mean(dim=2, keepdim=True)
            b = b - b.mean(dim=2, keepdim=True)
        num = (a * b).sum(dim=2)
        den = torch.sqrt((a * a).sum(dim=2) * (b * b).sum(dim=2) + 1e-5)
        return 1.0 - (num / den).mean()

    def loss(self, y_true, y_pred):
        fI_list = self._extract(y_true)
        fJ_list = self._extract(y_pred)
        per_layer = [self._per_layer_loss(a, b) for a, b in zip(fI_list, fJ_list)]
        return torch.stack(per_layer).mean()


class MIND:
    """Modality Independent Neighbourhood Descriptor (Heinrich et al., 2012).

    For each image, compute a self-similarity descriptor: at every pixel p and
    for a fixed set of neighbor offsets r, the descriptor value is
        D(p, r) = exp(-SSD_patch(I(p), I(p + r)) / V(p))
    where V(p) is a local variance estimate (mean SSD over neighbors). The loss
    is the L1 distance between the descriptors of the two images. Robust to
    intensity/contrast differences — useful for multi-modal registration.

    Uses a 4-neighborhood in 2D (offsets ±radius along each axis) and patch
    averaging with a box kernel.
    """

    def __init__(self, patch_size=3, radius=2):
        self.patch_size = patch_size
        self.radius = radius
        self.shifts = [(0, radius), (0, -radius), (radius, 0), (-radius, 0)]

    def _patch_avg(self, img):
        ps = self.patch_size
        kernel = torch.ones(1, 1, ps, ps, device=img.device, dtype=img.dtype) / (ps * ps)
        return F.conv2d(img, kernel, padding=ps // 2)

    def _descriptor(self, img):
        dists = []
        for dy, dx in self.shifts:
            shifted = torch.roll(img, shifts=(dy, dx), dims=(-2, -1))
            d = self._patch_avg((img - shifted) ** 2)
            dists.append(d)
        dists = torch.cat(dists, dim=1)  # [B, K, H, W]
        V = dists.mean(dim=1, keepdim=True).clamp(min=1e-6)
        mind = torch.exp(-dists / V)
        mind = mind / (mind.max(dim=1, keepdim=True)[0] + 1e-8)
        return mind

    def loss(self, y_true, y_pred):
        return (self._descriptor(y_true) - self._descriptor(y_pred)).abs().mean()


class NGF:
    """Normalized Gradient Fields loss (Haber & Modersitzki, 2006).

    Aligns the directions of image gradients regardless of magnitude or sign —
    good for handling bias fields and multi-modal intensity relationships.
    Returns mean(1 - <n_I, n_J>^2), which is 0 when gradients are aligned or
    anti-aligned everywhere, and 1 when they're orthogonal.
    """

    def __init__(self, eps=1e-5):
        self.eps = eps

    @staticmethod
    def _gradient(img):
        gy = img[..., 1:, :] - img[..., :-1, :]
        gx = img[..., :, 1:] - img[..., :, :-1]
        gy = F.pad(gy, (0, 0, 0, 1), mode='replicate')
        gx = F.pad(gx, (0, 1, 0, 0), mode='replicate')
        return gx, gy

    def loss(self, y_true, y_pred):
        Ix, Iy = self._gradient(y_true)
        Jx, Jy = self._gradient(y_pred)

        mag_I = torch.sqrt(Ix * Ix + Iy * Iy + self.eps ** 2)
        mag_J = torch.sqrt(Jx * Jx + Jy * Jy + self.eps ** 2)
        nIx, nIy = Ix / mag_I, Iy / mag_I
        nJx, nJy = Jx / mag_J, Jy / mag_J

        inner = nIx * nJx + nIy * nJy  # cos(theta), in [-1, 1]
        return (1.0 - inner * inner).mean()


class LTMALoss:
    """
    Original combined loss function (kept for backward compatibility).
    """

    def __init__(self, lambda_sim=1, lambda_reg=0.1, lambda_scale=0.001, alpha_scale=0.001,
                 lambda_low_structure=0.01, use_ncc=False, ncc_win=9,
                 sim_type=None, ssim_win=11, mi_bins=32, gamma=0.0001/2):
        self.lambda_sim = lambda_sim
        self.lambda_reg = lambda_reg
        self.lambda_scale = lambda_scale
        self.alpha_scale = alpha_scale
        self.lambda_low_structure = lambda_low_structure
        self.gamma = gamma

        # sim_type takes precedence; fall back to use_ncc for backward compat
        if sim_type is None:
            sim_type = 'ncc' if use_ncc else 'mse'
        if sim_type not in ('mse', 'l1', 'ncc', 'gncc', 'ssim', 'msssim',
                             'mi', 'nmi', 'ngf', 'mind', 'deepsim', 'deepsim_mse'):
            raise ValueError(f"Unknown sim_type: {sim_type}")
        self.sim_type = sim_type
        self.use_ncc = (sim_type == 'ncc')

        self.grad_loss = Grad(penalty='l2')
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ncc_loss = NCC(win=ncc_win)
        self.gncc_loss = GlobalNCC()
        self.ssim_loss = SSIM(win=ssim_win)
        self.msssim_loss = MS_SSIM(win=ssim_win)
        self.mi_loss = MI(num_bins=mi_bins)
        self.nmi_loss = NMI(num_bins=mi_bins)
        self.ngf_loss = NGF()
        self.mind_loss = MIND()
        self.deepsim_loss = DeepSim(similarity='ncc')
        self.deepsim_mse_loss = DeepSim(similarity='mse')

    def _sim(self, target, warped):
        if self.sim_type == 'mse':
            return self.mse_loss(warped, target)
        if self.sim_type == 'l1':
            return self.l1_loss(warped, target)
        if self.sim_type == 'ncc':
            return self.ncc_loss.loss(target, warped)
        if self.sim_type == 'gncc':
            return self.gncc_loss.loss(target, warped)
        if self.sim_type == 'ssim':
            return self.ssim_loss.loss(target, warped)
        if self.sim_type == 'msssim':
            return self.msssim_loss.loss(target, warped)
        if self.sim_type == 'mi':
            return self.mi_loss.loss(target, warped)
        if self.sim_type == 'nmi':
            return self.nmi_loss.loss(target, warped)
        if self.sim_type == 'ngf':
            return self.ngf_loss.loss(target, warped)
        if self.sim_type == 'mind':
            return self.mind_loss.loss(target, warped)
        if self.sim_type == 'deepsim':
            return self.deepsim_loss.loss(target, warped)
        if self.sim_type == 'deepsim_mse':
            return self.deepsim_mse_loss.loss(target, warped)
        raise ValueError(f"Unknown sim_type: {self.sim_type}")

    def compute_loss(self, source_img, target_imgs, warped_imgs, velocities, scaling_factors,
                     delta_tilde_per_frame=None, v_t_list=None, v_reg_list=None,
                     num_cardiac_frames=None):
        loss_dict = {}

        sim_loss = 0.0
        for t, (target, warped) in enumerate(zip(target_imgs, warped_imgs)):
            sim_loss += self._sim(target, warped)
        sim_loss /= len(target_imgs)
        loss_dict['similarity'] = sim_loss.item()

        reg_loss = 0.0
        for v in velocities:
            reg_loss += self.grad_loss.loss(v)
        reg_loss /= len(velocities)
        loss_dict['regularity'] = reg_loss.item()

        # Bridge log-likelihood using transition probability:
        #   v_t | v_{t-1} ~ N(mu_t, sigma_t^2)
        #   mu_t = v^r_t + ratio * (v_{t-1} - v^r_{t-1})
        #   sigma_t^2 = s_t * ratio, where ratio = (T-t)/(T-(t-1))
        bridge_norm = 0.0
        n_frames = 0
        T = num_cardiac_frames - 1  # last frame index
        for frame_idx in range(1, num_cardiac_frames - 1):  # interior frames only
            s_t = scaling_factors[:, frame_idx - 1]  # [B, 2, H, W]

            # Transition probability variance: s_t * (T - t) / (T - (t-1))
            ratio = (T - frame_idx) / (T - (frame_idx - 1))
            delta_t = s_t * ratio
            delta_t = torch.clamp(delta_t, min=1e-6)

            # Transition probability mean: v^r_t + ratio * (v_{t-1} - v^r_{t-1})
            mean_t = normalize_velocity(v_reg_list[frame_idx]) + ratio * (
                normalize_velocity(v_t_list[frame_idx - 1]) - normalize_velocity(v_reg_list[frame_idx - 1])
            )
            v_tilde = normalize_velocity(v_t_list[frame_idx]) - mean_t
            quadratic = torch.mean(v_tilde**2 / (delta_t))

            log_norm = torch.mean(0.5*torch.log(delta_t))

            bridge_norm += quadratic + log_norm
            n_frames += 1

        bridge_norm = bridge_norm / n_frames

        loss_dict['bridge_norm'] = bridge_norm.item() if isinstance(bridge_norm, torch.Tensor) else bridge_norm

        eps = 1e-6
        inverse_scale_loss = torch.mean(1.0 / (scaling_factors))
        loss_dict['inverse_scale'] = inverse_scale_loss.item()

        log_scale = torch.mean(torch.log(scaling_factors))
        inv_gamma_prior = self.lambda_scale * inverse_scale_loss + (self.alpha_scale+1) * log_scale
        loss_dict['inv_gamma_prior'] = inv_gamma_prior.item()

        # # --- Unified conjugate posterior terms ---
        # eps = 1e-6
        # n_interior = num_cardiac_frames - 2

        # residual_sum = 0.0
        # for frame_idx in range(1, num_cardiac_frames - 1):
        #     s_t = scaling_factors[:, frame_idx - 1]

        #     m_t = frame_idx / (num_cardiac_frames - 1)
        #     schedule_weight = m_t * (1.0 - m_t)

        #     v_tilde = v_t_list[frame_idx] - v_reg_list[frame_idx]

        #     residual_sum += torch.mean(v_tilde**2 / (4.0 * schedule_weight + eps))

        # beta_tilde = self.lambda_scale + residual_sum
        # alpha_tilde = (self.alpha_scale + 1) + n_interior / 2.0

        # inverse_scale = torch.mean(1.0 / (scaling_factors + eps))
        # log_scale = torch.mean(torch.log(scaling_factors + eps))

        # conjugate_loss = beta_tilde * inverse_scale + alpha_tilde * log_scale
        # loss_dict['inv_gamma_prior'] = conjugate_loss.item()

        #total_loss = sim_loss + (0.0001/2)*inv_gamma_prior + (0.0001/2)*bridge_norm
        total_loss = sim_loss + self.gamma*(inv_gamma_prior + bridge_norm + reg_loss)

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class EMLoss:
    """
    EM M-step loss based on the log joint posterior from BridgeUQ.

    The M-step maximizes the expected log posterior:
        L(theta) = E_v [sum_t -1/gamma * E(v_t(s_theta))
                        - |Omega|/2 * log(2*pi*delta_tilde_t(theta))
                        - beta / s_theta - (alpha+1) * log(s_theta)]

    After reparameterization (epsilon fixed from E-step), the ||epsilon||^2 term
    drops out, leaving three components to minimize:

    1. Registration energy:   1/gamma * E(v_t)
    2. Bridge normalization:  |Omega|/2 * log(delta_tilde_t)
    3. Inverse gamma prior:   beta / s + (alpha+1) * log(s)
    """

    def __init__(self, gamma=1.0, alpha=0.0001, beta=0.5,
                 bridge_norm_weight=0.5, use_ncc=False, ncc_win=9,
                 sim_type=None, ssim_win=11, mi_bins=32):
        """
        Args:
            gamma: Temperature parameter for likelihood (higher = less data fitting)
            alpha: Inverse gamma shape parameter (alpha+1 scales log(s))
            beta: Inverse gamma rate parameter (scales 1/s)
            bridge_norm_weight: Weight for the bridge normalization term
            use_ncc: [deprecated] Use NCC. Prefer sim_type='ncc'.
            ncc_win: Window size for local NCC
            sim_type: 'mse' | 'l1' | 'ncc' | 'gncc' | 'ssim'
            ssim_win: Window size for SSIM
        """
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.bridge_norm_weight = bridge_norm_weight

        if sim_type is None:
            sim_type = 'ncc' if use_ncc else 'mse'
        if sim_type not in ('mse', 'l1', 'ncc', 'gncc', 'ssim', 'msssim',
                             'mi', 'nmi', 'ngf', 'mind', 'deepsim', 'deepsim_mse'):
            raise ValueError(f"Unknown sim_type: {sim_type}")
        self.sim_type = sim_type
        self.use_ncc = (sim_type == 'ncc')

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ncc_loss = NCC(win=ncc_win)
        self.gncc_loss = GlobalNCC()
        self.ssim_loss = SSIM(win=ssim_win)
        self.msssim_loss = MS_SSIM(win=ssim_win)
        self.mi_loss = MI(num_bins=mi_bins)
        self.nmi_loss = NMI(num_bins=mi_bins)
        self.ngf_loss = NGF()
        self.mind_loss = MIND()
        self.deepsim_loss = DeepSim(similarity='ncc')
        self.deepsim_mse_loss = DeepSim(similarity='mse')
        self.grad_loss = Grad(penalty='l2')

    def _sim(self, target, warped):
        if self.sim_type == 'mse':
            return self.mse_loss(warped, target)
        if self.sim_type == 'l1':
            return self.l1_loss(warped, target)
        if self.sim_type == 'ncc':
            return self.ncc_loss.loss(target, warped)
        if self.sim_type == 'gncc':
            return self.gncc_loss.loss(target, warped)
        if self.sim_type == 'ssim':
            return self.ssim_loss.loss(target, warped)
        if self.sim_type == 'msssim':
            return self.msssim_loss.loss(target, warped)
        if self.sim_type == 'mi':
            return self.mi_loss.loss(target, warped)
        if self.sim_type == 'nmi':
            return self.nmi_loss.loss(target, warped)
        if self.sim_type == 'ngf':
            return self.ngf_loss.loss(target, warped)
        if self.sim_type == 'mind':
            return self.mind_loss.loss(target, warped)
        if self.sim_type == 'deepsim':
            return self.deepsim_loss.loss(target, warped)
        if self.sim_type == 'deepsim_mse':
            return self.deepsim_mse_loss.loss(target, warped)
        raise ValueError(f"Unknown sim_type: {self.sim_type}")

    def compute_m_step_loss(self, source_img, target_imgs, warped_imgs,
                            scaling_factors, delta_tilde_per_frame,
                            v_t_list=None, v_reg_list=None, num_cardiac_frames=None):
        """
        Compute the M-step loss for EM optimization.

        Args:
            source_img: Source image [B, 1, H, W]
            target_imgs: List of target images [I_1, ..., I_T], each [B, 1, H, W]
            warped_imgs: List of warped images from M-step forward pass, each [B, 1, H, W]
            scaling_factors: [B, T, 2, H, W] predicted scaling factors (without t=0)
            delta_tilde_per_frame: Dict mapping frame_idx -> delta_tilde [B, 2, H, W]
            v_t_list: List of sampled velocities [v_0, ..., v_T] from posterior SDE
            v_reg_list: List of registered velocities [v_0^r, ..., v_T^r]
            num_cardiac_frames: Total number of cardiac frames (T+1)

        Returns:
            total_loss, loss_dict
        """
        loss_dict = {}

        # 1. Registration energy with temperature: (1/gamma) * E(v_t)
        # E(v_t) = sum_t MSE(warp(source, v_t), target_t) / T
        energy_loss = 0.0
        for target, warped in zip(target_imgs, warped_imgs):
            energy_loss += self._sim(target, warped)
        energy_loss = energy_loss / len(target_imgs)
        scaled_energy = energy_loss
        loss_dict['energy'] = energy_loss.item()
        loss_dict['scaled_energy'] = scaled_energy.item()

        # 2. Bridge log-likelihood: -1/(2*delta_t)*||v_t - v_t^r||^2 - 1/2*log(2*pi*delta_t)
        #    where delta_t = 2*s*m_t*(1-m_t) is the marginal variance from the forward process
        #    and v_t is the sampled velocity from the posterior reverse SDE
        bridge_norm = 0.0
        n_frames = 0
        for frame_idx in range(1, num_cardiac_frames - 1):  # interior frames only
            # Scaling factor at this frame (scaling_factors excludes t=0)
            s_t = scaling_factors[:, frame_idx - 1]  # [B, 2, H, W]

            # Marginal variance: delta_t = 2*s*m_t*(1-m_t)
            m_t = frame_idx / (num_cardiac_frames - 1)
            delta_t = 2.0 * s_t * m_t * (1.0 - m_t)
            delta_t = torch.clamp(delta_t, min=1e-10)

            # Perturbation: v_tilde = v_t - v_t^r (normalized to [0,1])
            v_tilde = normalize_velocity(v_t_list[frame_idx]) - normalize_velocity(v_reg_list[frame_idx])

            # Huber-style quadratic: log(cosh(v_tilde / sqrt(delta_t)))
            ratio = v_tilde / torch.sqrt(delta_t)
            abs_ratio = torch.abs(ratio)
            quadratic = torch.mean(abs_ratio + torch.log1p(torch.exp(-2.0 * abs_ratio)))

            # # Old quadratic (unbounded):
            # quadratic = torch.mean(v_tilde**2 / (2.0 * delta_t))

            # Log normalization: 1/2 * log(2*pi*delta_t)
            log_norm = torch.mean(0.5 * torch.log(2.0 * np.pi * delta_t))

            bridge_norm += quadratic + log_norm
            n_frames += 1

        if n_frames > 0:
            bridge_norm = bridge_norm / n_frames

        # # Old bridge norm: only log-normalization with conditional variance delta_tilde
        # bridge_norm = 0.0
        # n_frames = 0
        # for frame_idx, delta_t in delta_tilde_per_frame.items():
        #     if frame_idx > 0:  # Skip t=0 (zero variance endpoint)
        #         log_2pi_delta = torch.log(2.0 * np.pi * torch.clamp(delta_t, min=1e-10))
        #         bridge_norm += torch.mean(log_2pi_delta)
        #         n_frames += 1
        # if n_frames > 0:
        #     bridge_norm = bridge_norm / n_frames

        loss_dict['bridge_norm'] = bridge_norm.item() if isinstance(bridge_norm, torch.Tensor) else bridge_norm

        # 3. Inverse gamma prior: beta / s + (alpha+1) * log(s)
        eps = 1e-6
        inv_s = torch.mean(1.0 / (scaling_factors + eps))
        log_s = torch.mean(torch.log(scaling_factors + eps))
        inv_gamma_prior = self.beta * inv_s + self.bridge_norm_weight* (self.alpha + 1) * log_s
        loss_dict['inv_gamma_prior'] = inv_gamma_prior.item()

        # Total M-step loss (minimize negative log posterior)
        total_loss = scaled_energy + self.gamma*inv_gamma_prior

        loss_dict['total'] = total_loss.item()

        # Additional diagnostics
        loss_dict['similarity'] = energy_loss.item()
        loss_dict['inverse_scale'] = inv_s.item()

        return total_loss, loss_dict

    def compute_loss(self, source_img, target_imgs, warped_imgs, velocities,
                     scaling_factors, delta_tilde_per_frame=None,
                     v_t_list=None, v_reg_list=None, num_cardiac_frames=None):
        """
        Unified interface compatible with both standard training and EM.

        If delta_tilde_per_frame is provided, uses the full EM M-step loss.
        Otherwise, falls back to a simplified loss (energy + inverse gamma prior).
        """
        if delta_tilde_per_frame is not None:
            return self.compute_m_step_loss(
                source_img, target_imgs, warped_imgs,
                scaling_factors, delta_tilde_per_frame,
                v_t_list=v_t_list, v_reg_list=v_reg_list,
                num_cardiac_frames=num_cardiac_frames
            )

        # Fallback: simplified loss without bridge normalization
        loss_dict = {}

        energy_loss = 0.0
        for target, warped in zip(target_imgs, warped_imgs):
            energy_loss += self._sim(target, warped)
        energy_loss = energy_loss / len(target_imgs)
        scaled_energy = energy_loss / self.gamma
        loss_dict['energy'] = energy_loss.item()
        loss_dict['similarity'] = energy_loss.item()
        loss_dict['scaled_energy'] = scaled_energy.item()

        eps = 1e-6
        inv_s = torch.mean(1.0 / (scaling_factors + eps))
        log_s = torch.mean(torch.log(scaling_factors + eps))
        inv_gamma_prior = self.beta * inv_s + (self.alpha + 1) * log_s
        loss_dict['inv_gamma_prior'] = inv_gamma_prior.item()
        loss_dict['inverse_scale'] = inv_s.item()

        total_loss = scaled_energy + inv_gamma_prior

        loss_dict['total'] = total_loss.item()
        loss_dict['bridge_norm'] = 0.0
        loss_dict['regularity'] = 0.0

        return total_loss, loss_dict
