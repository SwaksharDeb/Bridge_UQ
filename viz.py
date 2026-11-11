"""
Sigma visualization.

Shows sqrt(s0^2 + s1^2) of the scaling factors handed to the SDE, i.e. the same
quantity as row 4 of v3_iclr's training figure (trainer.py:894-900).

Deliberately does NOT underlay the source frame. trainer.py's version draws the
grayscale image and then blends sigma over it with alpha=0.7:

    imshow(src, cmap='gray'); imshow(s_mag, cmap='viridis', alpha=0.7)

which is misleading: a completely FLAT sigma still shows crisp anatomy in that
panel, because what is visible is the underlay bleeding through the blend, not
structure in sigma. That mistake was made repeatedly while measuring the
closed-form sigma, which is flat to 0.04% yet looked anatomical when overlaid.
Here sigma is drawn opaque and alone, so the colours are sigma and nothing else.

The closed-form s is time-invariant (one [2, H, W] field per sequence,
broadcast across frames by _assemble_scaling_factors_bb), exactly as the
learned s is, so a single panel per sequence is the whole story -- no need for
the 15-column strip.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_sigma_panel(s_raw, source_img=None, save_path=None, subject_ids=None,
                     max_subjects=8, vmin=None, vmax=None, title=None,
                     per_panel=False, overlay=False):
    """
    Render sqrt(s0^2 + s1^2), one panel per sequence. Sigma only -- no underlay.

    Args:
        s_raw:      [B, 2, H, W] the scaling factors actually handed to the SDE
                    (i.e. after the convention conversion), on any device.
        source_img: [B, 1, H, W] source frames. Drawn only with overlay=True.
        overlay:    True draws sigma the way v3_iclr's training figure does
                    (trainer.visualize_training_step, row '|s| (scale)'): the
                    source frame in gray, |s| in viridis at alpha 0.7 on top.
                    CAUTION (see module docstring): the underlay shows through,
                    so even a FLAT sigma looks anatomical. Judge structure from
                    overlay=False or from the numbers, not from this view.
        save_path:  output .png path.
        vmin/vmax:  colour limits. Default: shared across all panels, from the
                    data. Pass 0.4/1.5 to compare against the reference on its
                    own scale.
        per_panel:  True gives every panel its own 1-99 percentile colour range
                    (vmin/vmax ignored). With a shared scale one subject's
                    outlier (e.g. 3.7 against everyone else's 0.61-0.75) pushes
                    all other panels into the bottom few percent of the colour
                    bar and they look flat. The panel title still reports the
                    true min-max.

    Returns the (vmin, vmax) actually used, plus per-subject magnitude stats.
    """
    s = s_raw.detach().float().cpu().numpy()
    B = min(int(s.shape[0]), int(max_subjects))

    mags = [np.sqrt(s[i, 0] ** 2 + s[i, 1] ** 2) for i in range(s.shape[0])]
    if vmin is None:
        vmin = float(min(m.min() for m in mags))
    if vmax is None:
        vmax = float(max(m.max() for m in mags))

    ncol = min(B, 4)
    nrow = int(np.ceil(B / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.4 * nrow),
                             squeeze=False)

    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        ax.axis('off')
        if i >= B:
            continue
        if per_panel:
            lo, hi = np.percentile(mags[i], [1, 99])
        else:
            lo, hi = vmin, vmax
        if overlay and source_img is not None:
            ax.imshow(source_img[i, 0].detach().float().cpu().numpy(),
                      cmap='gray', vmin=0, vmax=1)
            im = ax.imshow(mags[i], cmap='viridis', vmin=lo, vmax=hi, alpha=0.7)
        else:
            im = ax.imshow(mags[i], cmap='viridis', vmin=lo, vmax=hi)
        name = (subject_ids[i] if subject_ids is not None and i < len(subject_ids)
                else f"subj {i}")
        ax.set_title(f"{name}\n|s| {mags[i].min():.2f}-{mags[i].max():.2f}",
                     fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(title or "Closed-form sigma  |s| = sqrt(s0^2 + s1^2)",
                 fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close()

    stats = {
        'vmin': vmin,
        'vmax': vmax,
        'mag_mean': float(np.mean([m.mean() for m in mags])),
        'mag_min': float(min(m.min() for m in mags)),
        'mag_max': float(max(m.max() for m in mags)),
        # Spatial contrast: a flat sigma has ~0 std, the reference has clear
        # anatomical structure.
        'mag_std_within': float(np.mean([m.std() for m in mags])),
    }
    return stats


def _box_mean(x, r):
    """Mean over a (2r+1)^2 window, replicate-padded, same size out."""
    import torch.nn.functional as F
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode='replicate'),
                        kernel_size=2 * r + 1, stride=1)


def guided_smooth_sigma(s_raw, source_img, radius=3, eps=1e-3):
    """
    Guided filter (He, Sun & Tang 2013) of s [B, C, H, W] with the source frame
    [B, 1, H, W] as guide, rescaled to [0, 1] per subject. Averages freely where
    the image is flat (local var << eps) and keeps s's transitions where the
    image has an edge (var >> eps). The slope is cov(I, log s) / (var I + eps),
    so structure in s that does not co-vary with the image is smoothed away, not
    copied in from the guide. Run on log s and exponentiated back, so the output
    stays positive (the filter's local linear model can undershoot zero on s
    itself). Visualization only -- the EM iteration never sees it.
    """
    r = max(1, int(radius))
    s = s_raw.detach().float()
    I = source_img.detach().float().to(s.device)
    lo = I.amin(dim=(1, 2, 3), keepdim=True)
    hi = I.amax(dim=(1, 2, 3), keepdim=True)
    I = (I - lo) / (hi - lo).clamp_min(1e-8)
    p = s.clamp_min(1e-12).log()
    mI, mp = _box_mean(I, r), _box_mean(p, r)
    a = (_box_mean(I * p, r) - mI * mp) / (_box_mean(I * I, r) - mI * mI + eps)
    b = mp - a * mI
    return (_box_mean(a, r) * I + _box_mean(b, r)).exp()


def save_sigma_smooth_panel(s_raw, source_img, save_path, subject_ids=None,
                            radius=3, eps=1e-3, max_subjects=8, title=None):
    """
    One row per sequence: source frame | raw |s| | guided-smoothed |s|, both
    |s| panels plain (no underlay) with their own 1-99 percentile colour range,
    as in sigma_post_smooth/batch_000_two_subjects.png.
    """
    sm = guided_smooth_sigma(s_raw, source_img, radius, eps)
    mag = lambda t: t.pow(2).sum(1).sqrt().cpu().numpy()
    raw_m, sm_m = mag(s_raw.detach().float()), mag(sm)
    src = source_img.detach().float().cpu().numpy()
    B = min(int(raw_m.shape[0]), int(max_subjects))

    fig, axes = plt.subplots(B, 3, figsize=(3.4 * 3, 3.0 * B), squeeze=False)
    for i in range(B):
        name = (subject_ids[i] if subject_ids is not None and i < len(subject_ids)
                else f"subj {i}")
        axes[i][0].imshow(src[i, 0], cmap='gray')
        axes[i][0].set_title(f"{name} source", fontsize=9)
        for j, (m, lab) in enumerate(((raw_m[i], 'raw'),
                                      (sm_m[i], f'guided r={radius} eps={eps:g}')),
                                     start=1):
            lo, hi = np.percentile(m, [1, 99])
            im = axes[i][j].imshow(m, cmap='viridis', vmin=lo, vmax=hi)
            axes[i][j].set_title(f"{lab}\n|s| {m.min():.3f}-{m.max():.3f}", fontsize=9)
            plt.colorbar(im, ax=axes[i][j], fraction=0.046, pad=0.02)
        for ax in axes[i]:
            ax.axis('off')
    plt.suptitle(title or "sigma |s|: raw vs guided-smoothed (plain, no underlay)",
                 fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close()
    return sm
