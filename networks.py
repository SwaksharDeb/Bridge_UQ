"""
Network components for learned scaling factor prediction.

The ScalingFactorNetwork takes BOTH the image sequence AND the registration
velocity sequence as input.  Each modality is independently projected to a
shared channel dimension and then *added* element-wise before being fed into
a U-Net encoder–decoder.

Input:  source_img + target_imgs + v_reg_list
        images:     [B, 1 + T, H, W]
        velocities: [B, T * 2, H, W]
Output: scaling_factors  ->  [B, 2, H, W]  (shared across time)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """Convolutional block with LeakyReLU activation."""

    def __init__(self, ndims, in_channels, out_channels, kernel=3, stride=1, padding=1):
        super().__init__()
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernel, stride, padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.main(x))


class ScalingFactorNetwork(nn.Module):
    """
    Network to predict spatially varying scaling factors from both images
    and velocity fields.

    Instead of concatenating images and velocities along the channel axis,
    each modality is projected to a shared dimension ``proj_dim`` by a small
    conv stack and the two projections are **added** element-wise.  This
    forces the network to learn modality-specific features before fusion and
    makes the scaling factor more sensitive to the velocity field produced by
    each registration backbone.

    Architecture: projection-add fusion  ->  U-Net encoder-decoder
    """

    def __init__(self, num_time_steps, img_size=64, num_heads=8, proj_dim=32):
        super().__init__()

        self.num_time_steps = num_time_steps
        self.img_size = img_size
        ndims = 2

        # Channel counts for each modality
        img_channels = 1 + num_time_steps        # source (1) + targets (T)
        vel_channels = num_time_steps * 2         # velocity x,y per frame

        # --- Projection branches ---
        # Two-layer conv projections: raw channels -> proj_dim
        self.img_proj = nn.Sequential(
            ConvBlock(ndims, img_channels, proj_dim),
            ConvBlock(ndims, proj_dim, proj_dim),
        )
        self.vel_proj = nn.Sequential(
            ConvBlock(ndims, vel_channels, proj_dim),
            ConvBlock(ndims, proj_dim, proj_dim),
        )

        # Current-velocity branch (v^k from inner loop). Same channel count as v_reg.
        self.vel_cur_proj = nn.Sequential(
            ConvBlock(ndims, vel_channels, proj_dim),
            ConvBlock(ndims, proj_dim, proj_dim),
        )

        # Instance normalization on each branch before fusion
        # Ensures both modalities contribute at similar magnitude,
        # preserving spatial patterns while preventing one from dominating
        self.img_norm = nn.InstanceNorm2d(proj_dim)
        self.vel_norm = nn.InstanceNorm2d(proj_dim)
        self.vel_cur_norm = nn.InstanceNorm2d(proj_dim)

        # --- U-Net encoder-decoder (starts from proj_dim channels) ---
        self.enc_channels = [proj_dim, 32, 64, 128, 256]

        # Encoder
        self.encoder = nn.ModuleList()
        self.pooling = nn.ModuleList()

        for i in range(len(self.enc_channels) - 1):
            self.encoder.append(ConvBlock(ndims, self.enc_channels[i], self.enc_channels[i+1]))
            self.pooling.append(nn.MaxPool2d(2))

        # Bottleneck
        self.bottleneck = ConvBlock(ndims, self.enc_channels[-1], self.enc_channels[-1])

        # Decoder with skip connections
        self.decoder = nn.ModuleList()
        self.upsampling = nn.ModuleList()

        dec_in_channels = [
            self.enc_channels[-1] + self.enc_channels[-1],  # 256 + 256
            128 + self.enc_channels[-2],  # 128 + 128
            64 + self.enc_channels[-3],   # 64 + 64
            32 + self.enc_channels[-4],   # 32 + 32
        ]
        dec_out_channels = [128, 64, 32, 16]

        for i in range(len(dec_out_channels)):
            self.decoder.append(ConvBlock(ndims, dec_in_channels[i], dec_out_channels[i]))
            self.upsampling.append(nn.Upsample(scale_factor=2, mode='nearest'))

        # Final convolutions
        self.final_conv1 = ConvBlock(ndims, 16, 16)
        self.final_conv2 = ConvBlock(ndims, 16, 16)

        # Output layer: predict scaling factor for velocity components (shared across time)
        self.output_conv = nn.Conv2d(16, 2, kernel_size=3, padding=1)

        # Softplus to ensure positive scaling factors
        self.softplus = nn.Softplus(beta=0.5)

        # Initialize output to produce values around 1
        nn.init.zeros_(self.output_conv.weight)
        nn.init.constant_(self.output_conv.bias, 0.0)

    def forward(self, source_img, target_imgs, v_reg_list=None, v_current_list=None):
        """
        Args:
            source_img: [B, 1, H, W] source image
            target_imgs: List of [B, 1, H, W] target images, length T
                        OR stacked tensor [B, T, 1, H, W]
            v_reg_list: List of T velocity fields [B, 2, H, W] (the registered velocity)
                        OR pre-concatenated tensor [B, T*2, H, W]
            v_current_list: List/tensor of the current velocity v^k.
                        If None, falls back to v_reg_list (i.e. v^0 = v^r).

        Returns:
            scaling_factors: [B, 2, H, W] spatially varying scaling factors
        """
        if v_reg_list is None:
            raise ValueError("v_reg_list must be provided for combined scaling network")

        if v_current_list is None:
            v_current_list = v_reg_list

        # Handle target_imgs as list or tensor
        if isinstance(target_imgs, list):
            targets_cat = torch.cat(target_imgs, dim=1)  # [B, T, H, W]
        else:
            # Assume [B, T, 1, H, W]
            B, T, C, H, W = target_imgs.shape
            targets_cat = target_imgs.squeeze(2)  # [B, T, H, W]

        # Handle v_reg_list as list or pre-concatenated tensor
        if isinstance(v_reg_list, list):
            vel_cat = torch.cat(v_reg_list, dim=1)  # [B, T*2, H, W]
        else:
            vel_cat = v_reg_list  # Already [B, T*2, H, W]

        if isinstance(v_current_list, list):
            vel_cur_cat = torch.cat(v_current_list, dim=1)
        else:
            vel_cur_cat = v_current_list

        # --- Projection + instance-norm + additive fusion (3-way) ---
        img_input = torch.cat([source_img, targets_cat], dim=1)        # [B, 1+T, H, W]
        img_feat     = self.img_norm(self.img_proj(img_input))         # [B, proj_dim, H, W]
        vel_reg_feat = self.vel_norm(self.vel_proj(vel_cat))           # [B, proj_dim, H, W]
        vel_cur_feat = self.vel_cur_norm(self.vel_cur_proj(vel_cur_cat))
        x = img_feat + vel_reg_feat + vel_cur_feat                     # additive 3-way fusion

        return self.trunk(x)

    def trunk(self, x):
        """
        U-Net encoder -> bottleneck -> decoder -> positive [B, 2, H, W] output.

        Split out of forward() so subclasses can fuse a different set of input
        branches and still share this body verbatim -- see
        SigmaRegressionNetwork, which fuses two branches instead of three.
        """
        # Encoder with skip connection storage
        enc_features = []
        for conv, pool in zip(self.encoder, self.pooling):
            x = conv(x)
            enc_features.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections (reverse order)
        for i, (conv, upsample) in enumerate(zip(self.decoder, self.upsampling)):
            x = upsample(x)
            skip_idx = len(enc_features) - 1 - i
            if skip_idx >= 0:
                skip = enc_features[skip_idx]
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear')
                x = torch.cat([x, skip], dim=1)
            x = conv(x)

        # Final convolutions
        x = self.final_conv1(x)
        x = self.final_conv2(x)

        # Output scaling factors
        x = self.output_conv(x)  # [B, 2, H, W]

        # Ensure positive scaling factors
        x = self.softplus(x)

        return x


class SigmaRegressionNetwork(ScalingFactorNetwork):
    """
    Predict sigma from the image sequence and the REGISTERED velocity only.

    Same trunk as ScalingFactorNetwork, one branch fewer: there is no
    v_current input. ScalingFactorNetwork is used inside the EM loop, where
    v^k -- the current sampled velocity -- is available and carries most of the
    signal about how much the trajectory is moving. This network is meant to be
    used WITHOUT that loop, at test time on a new subject, where only the
    images and the backbone's v_reg exist. It is trained by supervised
    regression onto the sigma the EM loop converged to (see
    main_langevin.py --dump_sigma_targets and train_sigma_regressor.py).

    Dropping the v_current branch is the whole point, so it is deleted rather
    than left unused: an unused branch would still be constructed, saved into
    every checkpoint, and could silently be fed something by mistake.

    Output is [B, 2, H, W], positive via the same softplus, and in whichever
    units the targets were saved in -- `s_raw` or `sigma = 2s/T`. The training
    script records which, and `target_field` is stored in the checkpoint.
    """

    def __init__(self, num_time_steps, img_size=64, proj_dim=32,
                 init_output=None):
        super().__init__(num_time_steps, img_size=img_size, proj_dim=proj_dim)
        # The v_current branch has no input here.
        del self.vel_cur_proj
        del self.vel_cur_norm
        if init_output is not None:
            self.set_output_bias(float(init_output))

    def set_output_bias(self, value):
        """
        Bias the output layer so the network starts near `value`.

        output_conv's weights are zero-initialized, so at step 0 every voxel
        emits softplus(bias). With beta = 0.5 that is
        (1/beta) log(1 + exp(beta*b)), which inverts to
        b = (1/beta) log(exp(beta*v) - 1). Starting at the target mean rather
        than at softplus(0) = 1.386 matters when the targets are small -- the
        'paper_sigma' convention puts them near 0.09, and a 15x overshoot at
        init wastes the early epochs.
        """
        beta = self.softplus.beta
        v = max(float(value), 1e-6)
        b = float(np.log(np.expm1(beta * v)) / beta)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.constant_(self.output_conv.bias, b)

    def forward(self, source_img, target_imgs, v_reg_list):
        """
        Args:
            source_img:  [B, 1, H, W]
            target_imgs: list of T [B, 1, H, W], or [B, T, 1, H, W]
            v_reg_list:  list of T [B, 2, H, W], or [B, T*2, H, W]

        Returns:
            [B, 2, H, W] predicted sigma (positive).
        """
        if isinstance(target_imgs, list):
            targets_cat = torch.cat(target_imgs, dim=1)
        else:
            targets_cat = target_imgs.squeeze(2)

        vel_cat = (torch.cat(v_reg_list, dim=1) if isinstance(v_reg_list, list)
                   else v_reg_list)

        img_input = torch.cat([source_img, targets_cat], dim=1)
        img_feat = self.img_norm(self.img_proj(img_input))
        vel_feat = self.vel_norm(self.vel_proj(vel_cat))
        x = img_feat + vel_feat                     # 2-way additive fusion

        return self.trunk(x)
