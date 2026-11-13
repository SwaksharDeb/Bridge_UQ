import torch
import torch.nn as nn


# lp_loss = LpRelMSELoss(W_Mse=5, W_Rel=5, p=2)
# combined, parts = lp_loss(x_seq, y_seq)  # x_seq,y_seq: [B,T,C,H,W]

class LpRelMSELoss(nn.Module):
    """
    Combined loss: W_Rel * ||x - y||_p / ||y||_p  +  W_Mse * MSE, averaged over (B,T).
    Accepts [B, T, C, ...] or list(T) of [B, C, ...].
    """
    def __init__(self, W_Mse: float = 5.0, W_Rel: float = 5.0, p: int = 2, eps: float = 1e-12):
        super().__init__()
        self.W_Mse = W_Mse
        self.W_Rel = W_Rel
        self.p = p
        self.eps = eps

    def forward(self, x, y):
        # unify to [BT, ...]
        if isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            x = torch.stack(x, dim=1)  # [B, T, C, ...]
            y = torch.stack(y, dim=1)
        assert x.shape == y.shape, "x/y shape mismatch"
        B, T = x.shape[:2]
        x_flat = x.reshape(B*T, -1)
        y_flat = y.reshape(B*T, -1)

        # relative Lp over samples
        diff_norm = torch.norm(x_flat - y_flat, p=self.p, dim=1)
        y_norm    = torch.norm(y_flat,          p=self.p, dim=1).clamp_min(self.eps)
        rel = (diff_norm / y_norm).mean()

        # MSE over samples
        mse = torch.mean((x_flat - y_flat)**2, dim=1).mean()

        return self.W_Rel * rel + self.W_Mse * mse, {"rel": rel, "mse": mse}




class MVMList(nn.Module):
    """
    Average over time of <v, m> / (#elements per vector component).
    V, M: [B, T, C, ...] or list(T) of [B, C, ...]; C=ndims.
    """
    def __init__(self, ndims: int = 2):
        super().__init__()
        self.ndims = ndims

    def forward(self, VList_Tensor, MList_Tensor):
        lenV = VList_Tensor.shape[0]
        nums = VList_Tensor[0].numel()/self.ndims
        VMList = [(VList_Tensor[i]*MList_Tensor[i]).sum() / nums for i in range(lenV)]
        # #print(VMList)
        return sum(VMList) / lenV

    def vm_l2_norm_per_sample(self, v, m, ndims=2):  # v,m: [B,C,...]
        dims = tuple(range(1, v.ndim))
        return (v*m).sum(dim=dims) / (v[0].numel() / ndims)

    def reparametrize_vec(self, V_list, M_list, ndims=2):
        # V_list/M_list: list(T) of [B,C,...]
        B = V_list[0].shape[0]; T = len(V_list)
        norms = [self.vm_l2_norm_per_sample(V_list[i], M_list[i], ndims) for i in range(T)]  # [T][B]
        mean_len = sum(norms) / T  # [B]
        if ndims == 2:
            view = (B, 1, 1, 1)
        else:
            view = (B, 1, 1, 1, 1)
        scales = [torch.sqrt(norms[i] / mean_len).view(view) for i in range(T)]
        return [V_list[i] / scales[i] for i in range(T)]


import torch.nn.functional as F
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NCC_vxm(nn.Module):
    """
    Local (over window) normalized cross correlation loss.

    Parameters
    ----------
    ndims : int
        Number of spatial dimensions (e.g., 2 for [B,1,H,W], 3 for [B,1,D,H,W]).
    win : int or None
        Window size. If None, defaults to 9.
    reduction : str
        'mean' or 'none', like PyTorch losses.
        - 'mean': return a scalar (global mean over all voxels and batch)
        - 'none': return the NCC map (same spatial shape as input)
    """

    def __init__(self, ndims=2, win=9, reduction='mean'):
        super().__init__()
        self.ndims = ndims
        self.win = win
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        Ii = y_true
        Ji = y_pred

        ndims = self.ndims
        win = [9] * ndims if self.win is None else [self.win] * ndims
        # sum_filt = torch.ones([1, 1, *win], device=Ii.device, dtype=Ii.dtype)

        B, C = Ii.shape[:2]
        sum_filt = torch.ones([C, 1, *win], device=Ii.device, dtype=Ii.dtype)


        pad_no = win[0] // 2
        stride = (1,) * ndims
        padding = (pad_no,) * ndims
        conv_fn = getattr(F, f'conv{ndims}d')

        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum  = conv_fn(Ii,  sum_filt, stride=stride, padding=padding, groups=C)
        J_sum  = conv_fn(Ji,  sum_filt, stride=stride, padding=padding, groups=C)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding, groups=C)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding, groups=C)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding, groups=C)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)  # NCC^2

        if self.reduction == 'none':
            # return NCC-based "loss map" (negative NCC^2), to be reduced outside
            return -cc
        else:
            # default: scalar loss
            return -torch.mean(cc)

    





class MVGradList(nn.Module):
    """
    Spatial smoothness over each frame, then mean over (B,T).
    penalty='l2' -> squared diffs (H¹); 'l1' -> TV.
    Accepts [B,T,C,H,W] (2D) or [B,T,C,D,H,W] (3D) or list(T) of [B,C,...].
    """
    def __init__(self, ndims: int = 2, penalty: str = 'l2'):
        super().__init__()
        self.ndims = ndims
        self.penalty = penalty


    def forward(self, y_pred):
        if(self.ndims == 3):
            dy = torch.abs(y_pred[..., 1:, :, :] - y_pred[..., :-1, :, :])
            dx = torch.abs(y_pred[..., :, 1:, :] - y_pred[..., :, :-1, :])
            dz = torch.abs(y_pred[..., :, :, 1:] - y_pred[..., :, :, :-1])

            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                dz = dz * dz
            d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
            grad = d / 3.0
        
        elif(self.ndims == 2):
            dy = torch.abs(y_pred[..., 1:, :] - y_pred[..., :-1, :])
            dx = torch.abs(y_pred[..., :, 1:] - y_pred[..., :, :-1])
        
            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
            d = torch.mean(dx) + torch.mean(dy)
            grad = d / 2.0

        return grad



class TemporalGradLoss(nn.Module):
    """
    Spatial smoothness over each frame, then mean over (B,T).
    penalty='l2' -> squared diffs (H¹); 'l1' -> TV.
    Accepts [B,T,C,H,W] (2D) or [B,T,C,D,H,W] (3D) or list(T) of [B,C,...].
    """
    def __init__(self, penalty: str = 'l2'):
        super().__init__()
        self.penalty = penalty


    def forward(self, y_pred): # [B,T,C,H,W] or [B,T,C,D,H,W]
        dt = torch.abs(y_pred[:, 1:] - y_pred[:, :-1])
        if self.penalty == 'l2': 
            dt = dt * dt
        d = torch.mean(dt)
        return d



class TemporalSpatialGradLoss(nn.Module):
    """
    Spatial smoothness over each frame, then mean over (B,T).
    penalty='l2' -> squared diffs (H¹); 'l1' -> TV.
    Accepts [B,T,C,H,W] (2D) or [B,T,C,D,H,W] (3D) or list(T) of [B,C,...].
    """
    def __init__(self, ndims: int = 2, penalty: str = 'l2'):
        super().__init__()
        self.ndims = ndims
        self.penalty = penalty


    def forward(self, y_pred): # [B,T,C,H,W] or [B,T,C,D,H,W]
        if(self.ndims == 3):
            dy = torch.abs(y_pred[..., 1:, :, :] - y_pred[..., :-1, :, :])
            dx = torch.abs(y_pred[..., :, 1:, :] - y_pred[..., :, :-1, :])
            dz = torch.abs(y_pred[..., :, :, 1:] - y_pred[..., :, :, :-1])
            dt = torch.abs(y_pred[..., 1:, :, :, :] - y_pred[..., :-1, :, :, :])

            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                dz = dz * dz
                dt = dt * dt
            d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz) + torch.mean(dt)
            grad = d / 4.0
        elif(self.ndims == 2):
            dy = torch.abs(y_pred[..., 1:, :] - y_pred[..., :-1, :])
            dx = torch.abs(y_pred[..., :, 1:] - y_pred[..., :, :-1])
            dt = torch.abs(y_pred[..., 1:, :, :] - y_pred[..., :-1, :, :])
          
            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                dt = dt * dt
            d = torch.mean(dx) + torch.mean(dy) + torch.mean(dt)
            grad = d / 3.0

        return grad



