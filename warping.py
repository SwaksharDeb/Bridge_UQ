"""
Image warping utilities using VecInt and SpatialTransformer.

This module provides warping functionality for images using velocity fields.
"""

import os
import sys

import torch
from utils.Int import VecInt, SpatialTransformer


# Global warping instances (shared across modules)
_vec_int = None
_spatial_transformer = None

# torch.compile support (opt-in via enable_compiled_warp, see main.py --compile).
# _warp_core holds the tensor-only part of the warp so it can be handed to
# torch.compile as a single graph; the eager path calls the same function.
_compiled_core = None


def _warp_core(image, velocity):
    """VecInt scaling-and-squaring + spatial transform. Tensors in, tensor out."""
    disp_list = _vec_int(velocity)
    displacement = disp_list[-1]
    warped, _ = _spatial_transformer(image, displacement)
    return warped


def enable_compiled_warp(mode='default', dynamic=False):
    """
    Route warp_image_with_velocity through torch.compile.

    The warp is the hot path of the whole pipeline: every posterior reverse-SDE
    step calls it once for the energy gradient (forward + backward through 7
    grid_sample steps), and the loss calls it once per cardiac frame. Eager mode
    launches ~40 small kernels per call, which is launch-bound at these tensor
    sizes; inductor fuses the elementwise grid math around each grid_sample.

    No-op (with a warning) if torch.compile is unavailable — e.g. torch 2.0 on
    Python 3.11, where dynamo raises "Python 3.11+ not yet supported".
    """
    global _compiled_core
    import torch as _torch
    try:
        _compiled_core = _torch.compile(_warp_core, mode=mode, dynamic=dynamic)
    except Exception as e:  # unsupported torch/python combination
        print(f"[warping] torch.compile unavailable ({type(e).__name__}: {e}); "
              f"falling back to eager warp.")
        _compiled_core = None
        return False
    return True


def disable_compiled_warp():
    """Restore the eager warp path."""
    global _compiled_core
    _compiled_core = None


def warp_image_with_velocity(image, velocity, inshape=(64, 64)):
    """
    Warp an image using a velocity field via VecInt and SpatialTransformer.

    Args:
        image: Input image tensor [B, C, H, W] or [C, H, W] or [H, W]
        velocity: Velocity field tensor [B, 2, H, W] or [2, H, W]
        inshape: Tuple of (H, W) for the spatial dimensions

    Returns:
        Warped image with same shape as input
    """
    global _vec_int, _spatial_transformer

    if _vec_int is None:
        _vec_int = VecInt(inshape, TSteps=7)
        _spatial_transformer = SpatialTransformer(inshape)

    # Move modules to same device as input
    device = velocity.device
    _vec_int.to(device)
    _spatial_transformer.to(device)

    squeeze_batch = False
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
        squeeze_batch = True
    elif image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_batch = True

    if velocity.dim() == 3:
        velocity = velocity.unsqueeze(0)

    core = _compiled_core if _compiled_core is not None else _warp_core
    warped = core(image, velocity)

    if squeeze_batch:
        warped = warped.squeeze(0)

    return warped


def displacement_from_velocity(velocity, inshape=(64, 64)):
    """
    Integrate a velocity field to the displacement field the warp samples with.

    This is the first half of `warp_image_with_velocity`: VecInt's
    scaling-and-squaring, returning phi - Id instead of consuming it. Exposed so
    diagnostics that need the deformation itself (the Jacobian determinant, in
    particular) get the SAME field the warp uses rather than re-integrating the
    velocity with their own settings.

    Units are PIXELS, matching SpatialTransformer's grid (`torch.arange(0, s)`),
    so phi(x) = x + u(x) has unit spacing and det(I + grad u) needs no rescaling.

    Args:
        velocity: [B, 2, H, W] or [2, H, W]
        inshape:  (H, W)

    Returns:
        Displacement [B, 2, H, W].
    """
    global _vec_int, _spatial_transformer

    if _vec_int is None:
        _vec_int = VecInt(inshape, TSteps=7)
        _spatial_transformer = SpatialTransformer(inshape)

    _vec_int.to(velocity.device)
    _spatial_transformer.to(velocity.device)

    if velocity.dim() == 3:
        velocity = velocity.unsqueeze(0)
    return _vec_int(velocity)[-1]


def reset_warping_modules():
    """Reset the global warping modules (useful for testing or changing image size)."""
    global _vec_int, _spatial_transformer
    _vec_int = None
    _spatial_transformer = None
