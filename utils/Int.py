import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as nnf
# import lagomorph as lm
# from lagomorph import adjrep
# from lagomorph import deform
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
import os
from typing import Callable, Dict, List, Optional
import torch.nn.functional as F
from functools import reduce
from operator import mul
from scipy.ndimage import distance_transform_edt, gaussian_filter



class LocalAffineSpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer with a local affine field.
    inshape: (H, W) or (D, H, W)
    local_affine: [B, *inshape, ndims, ndims+1]
        2D: [B, H, W, 2, 3]
        3D: [B, D, H, W, 3, 4]
    """

    def __init__(self, inshape, mode='bilinear'):
        super().__init__()

        self.inshape = tuple(inshape)
        self.ndims = len(inshape)
        self.mode = mode

        vectors = [torch.arange(0, s) for s in inshape]
        grids = torch.meshgrid(*vectors, indexing='ij')
        grid = torch.stack(grids, dim=0)      # [ndims, *inshape]
        grid = grid.unsqueeze(0).float()      # [1, ndims, *inshape]

        grid_norm = torch.zeros_like(grid)

        for i in range(len(inshape)):
            s = inshape[i]
            grid_norm[:, i, ...] = 2.0 * (grid[:, i, ...] / (s - 1) - 0.5)

        self.register_buffer('grid', grid)
        self.register_buffer('grid_norm', grid_norm)
        # print(grid.shape) #torch.Size([1, 2, 64, 64])
        # assert 4>333

        # #sava grid as npy file for visualization
        # np.save("/scratch/bsw3ac/nellie/code/cvpr/Project/LightningTemplate/models/aff_grid.npy", grid.cpu().numpy())
        # np.save("/scratch/bsw3ac/nellie/code/cvpr/Project/LightningTemplate/models/grid_norm.npy", grid_norm.cpu().numpy())
    def normalize_grid(self, grid, inshape):
        grid_norm = torch.zeros_like(grid)
        for i in range(len(inshape)):
            s = inshape[i]
            grid_norm[:, i, ...] = 2.0 * (grid[:, i, ...] / (s - 1) - 0.5)
        return grid_norm
    
    def denormalize_grid(self, grid_norm, inshape):
        grid = torch.zeros_like(grid_norm)
        for i in range(len(inshape)):
            s = inshape[i]
            grid[:, i, ...] = (grid_norm[:, i, ...] + 1.0) / 2.0 * (s - 1)
        return grid

    def get_affine_grid(self, local_affine):
        """
        Args:
            local_affine: [B, *inshape, ndims, ndims+1]

        Returns:
            grid_norm: [B, *inshape, ndims] in [-1, 1]
        """
        expected_dims = self.ndims + 3    # B + spatial(ndims) + (ndims, ndims+1)
        assert local_affine.dim() == expected_dims, \
            f"Expected {expected_dims}D local_affine, got {local_affine.dim()}D"

        B = local_affine.shape[0]
        spatial_shape = local_affine.shape[1:-2]
        assert spatial_shape == self.inshape, \
            f"local_affine spatial shape {spatial_shape} != inshape {self.inshape}"

        nd = local_affine.shape[-2]
        p  = local_affine.shape[-1]
        assert nd == self.ndims
        assert p  == self.ndims + 1

        # flatten spatial dims -> N
        N = reduce(mul, self.inshape, 1)
        # print(N) #16384
        local_affine_flat = local_affine.view(B, N, self.ndims, self.ndims + 1)

        device = local_affine.device
        dtype  = local_affine.dtype

        # 1. Identity grid in index coordinates: [1, ndims, *inshape]
        base_grid = self.grid_norm.to(device=device, dtype=dtype)
        # print(11, base_grid.shape)    #11 torch.Size([1, 2, 128, 128])

        # [1, ndims, N] -> [1, N, ndims] -> [B, N, ndims]
        base_flat = base_grid.view(1, self.ndims, -1).permute(0, 2, 1)
        # print(base_flat.shape)   #torch.Size([1, 16384, 2])
        base_flat = base_flat.expand(B, -1, -1)
        # print(22, base_flat.shape)     #22 torch.Size([1, 16384, 2])

        # 2. Homogeneous coordinates: [x, y, (z), 1]
        ones = torch.ones(B, N, 1, device=device, dtype=dtype)
        pts_h = torch.cat([base_flat, ones], dim=-1)  # [B, N, ndims+1]
        # print(33, pts_h.shape)         #33 torch.Size([1, 16384, 3])

        # 3. Apply local affine at each location:
        pts_out = torch.matmul(local_affine_flat, pts_h.unsqueeze(-1)).squeeze(-1)
        # pts_out: [B, N, ndims] in index coords
        # print(44, pts_out.shape)        #44 torch.Size([1, 16384, 2])

        # 4. Reshape to [B, ndims, *inshape] for per-dim normalization
        new_locs = pts_out.permute(0, 2, 1).reshape(B, self.ndims, *self.inshape)
        # new_locs = pts_out.reshape(B, self.ndims, *self.inshape)

        """ # 6. Return as [B, *inshape, ndims] for grid_sample
        grid_norm = new_locs.permute(0, *range(2, 2 + self.ndims), 1)  #self.ndims = 2，那就是 range(2, 4) → 生成序列：[2, 3]
        # print(66, grid_norm.shape)      #66 torch.Size([1, 128, 128, 2])  #66 torch.Size([6, 64, 64, 2])
        # assert 4>555
        grid_norm = grid_norm.permute(0,2,1,3) #this grid has different order from identity grid """


        grid_norm = new_locs.permute(0,2,3,1)   # [B,H,W,2] but in (y,x)
        grid_norm = grid_norm[..., [1,0]]            # (y,x) -> (x,y)

        
        return grid_norm

    def forward(self, src, local_affine, align_corners=False):
        """ grid = self.get_affine_grid(local_affine)
        warped = F.grid_sample(src, grid, mode=self.mode)
        grid_denorm = self.denormalize_grid(grid.permute(0,3,1,2), self.inshape)

        disp = grid_denorm.permute(0,1,3,2)
        disp =  disp - self.grid """
        
        
        

        grid_xy_norm = self.get_affine_grid(local_affine)   # [B,H,W,2] in (x,y), normalized
        warped = F.grid_sample(src, grid_xy_norm, mode=self.mode)
        # to [B,2,H,W] in (x,y)
        grid_xy_norm_ch = grid_xy_norm.permute(0,3,1,2)
        # denormalize -> pixel coords, still (x,y)
        grid_xy_pix = self.denormalize_grid(grid_xy_norm_ch, self.inshape)  # [B,2,H,W] (x,y) in pixels
        # convert to (y,x) to match self.grid (which is (y,x) from meshgrid ij)
        grid_yx_pix = grid_xy_pix[:, [1,0], ...]  # [B,2,H,W] (y,x)
        # disp in (dy, dx)
        disp = grid_yx_pix - self.grid  # self.grid is [1,2,H,W] (y,x)
        return warped, grid_xy_norm, disp
        return warped, grid, disp

    def deformWithNormedGrid(self, src, grid, align_corners=False, mode=None):
        if mode is None:
            mode = self.mode
        warped = F.grid_sample(src, grid, mode=mode)
        return warped

    
    def getU(self, local_affine, align_corners=False):
        grid = self.get_affine_grid(local_affine)
        grid_denorm = self.denormalize_grid(grid.permute(0,3,1,2), self.inshape) #self.grid torch.Size([1, 2, 64, 64])
        assert 4>444
        disp = grid_denorm - self.grid
        return grid, grid_denorm, disp





class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        if(len(y_pred.shape) == 5):
            dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
            dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
            dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                dz = dz * dz

            d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
            grad = d / 3.0

            if self.loss_mult is not None:
                grad *= self.loss_mult
            return grad
        elif(len(y_pred.shape) == 4):
            # #print("y_pred   ",y_pred.shape)
            dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
            dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
          
            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
            d = torch.mean(dx) + torch.mean(dy)
            grad = d / 2.0

            if self.loss_mult is not None:
                grad *= self.loss_mult
            return grad


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """
    def __init__(self, inshape, mode='bilinear'):
        super().__init__()

        self.mode = mode
        # create sampling grid
        vectors = [torch.arange(0, s) for s in inshape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.cuda.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)
        self.grid = grid

        # #print(self.grid.shape, inshape)
        # assert 4>766

    def forward(self, src, flow, mode=None):
        if mode is None:
            mode = self.mode
        # # new locations
        # #print("flow: ",flow.shape, self.grid.shape) #flow: ([120, 2, 32, 32]) ([1, 2, 32, 32])
        # assert 4>567

        new_locs = self.grid + flow   #self.grid:  identity

        
        new_locs_unnormalize = self.grid + flow
        shape = flow.shape[2:]
        #  new_locs  :  torch.Size([1, 3, 64, 64, 64])
        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

            new_locs_unnormalize = new_locs_unnormalize.permute(0, 2, 3, 1) #[1, 64, 64, 64,3]
            new_locs_unnormalize = new_locs_unnormalize[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

            new_locs_unnormalize = new_locs_unnormalize.permute(0, 2, 3, 4, 1)
            new_locs_unnormalize = new_locs_unnormalize[..., [2, 1, 0]]


        # #print(torch.max(new_locs), torch.min(new_locs))  #[-1,1]
        # #print(src.shape, new_locs.shape,"   11111")  #src: [1, 3, 64, 64] new_locs: [1, 2, 64, 64]
        # assert 2>345

        


        warped = nnf.grid_sample(src, new_locs, mode=mode)

        # return warped, new_locs.permute(0, 4, 1, 2, 3)
        return warped, flow
    
    def get_new_locs_unnormalize(self, flow):  #torch.Size([1, 2, 64, 64])
        return self.grid + flow



class SpatialTransformer2(nn.Module):
    """
    N-D Spatial Transformer
    """
    def __init__(self, inshape, mode='bilinear'):
        super().__init__()

        self.mode = mode
        # create sampling grid
        vectors = [torch.arange(0, s) for s in inshape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.cuda.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)
        self.grid = grid

    def forward(self, src, flow, dt=1.0, mode=None):
        # #print("flow: ",flow.shape, self.grid.shape) #flow: ([120, 2, 32, 32]) ([1, 2, 32, 32])
        #print device
        # #print("device: ",flow.device, self.grid.device)
        
        flow = flow * dt
        if mode is None:
            mode = self.mode
        # new locations
        # #print("flow: ", flow.shape, self.grid.shape, "______________________________________")  # flow: ([1, 2, 64, 64]) ([1, 2, 64, 64])
        # assert 4 > 567
        new_locs = self.grid + flow   #self.grid:  identity
        # new_locs_unnormalize = self.grid + flow
        shape = flow.shape[2:]
        #  new_locs  :  torch.Size([1, 3, 64, 64, 64])
        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

            # new_locs_unnormalize = new_locs_unnormalize.permute(0, 2, 3, 1) #[1, 64, 64, 64,3]
            # new_locs_unnormalize = new_locs_unnormalize[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

            # new_locs_unnormalize = new_locs_unnormalize.permute(0, 2, 3, 4, 1)
            # new_locs_unnormalize = new_locs_unnormalize[..., [2, 1, 0]]

        warped = nnf.grid_sample(src, new_locs, mode=mode)
       
        return warped
    
    def get_new_locs_unnormalize(self, flow):  #torch.Size([1, 2, 64, 64])
        return self.grid + flow




class Epdiff():
    def __init__(self,inshape,alpha=2.0,gamma=1.0, TSteps=7):
        # alpha=2.0;gamma = 1.0
        self.nsteps = TSteps
        fluid_params = [alpha, 0, gamma]; 
        self.metric = lm.FluidMetric(fluid_params)
        self.grid = self.identity(inshape)
        self.transformer = SpatialTransformer2(inshape)
        



    def identity(self, inshape, dtype=np.float32):
        """
        Given a deformation shape in NCWH(D) order, produce an identity matrix (numpy array)
        """
        if len(inshape) == 2:
            defshape = (1,2,inshape[0],inshape[1])
        elif len(inshape) == 3:
            defshape = (1,3,inshape[0],inshape[1],inshape[2])

        dim = len(defshape)-2
        ix = np.empty(defshape, dtype=dtype)
        for d in range(dim):
            ld = defshape[d+2]
            shd = [1]*len(defshape)
            shd[d+2] = ld
            ix[:,d,...] = np.arange(ld, dtype=dtype).reshape(shd)
        return ix

    def EPDiff_step(self, m0, dt, phiinv, mommask=None):
        m = adjrep.Ad_star(phiinv, m0)
        if mommask is not None:
            m = m * mommask
        v = self.metric.sharp(m)
        return deform.compose_disp_vel(phiinv, v, dt=-dt), m, v
    
    def ShootWithV0(self, v0):
        m0 = self.metric.flat(v0)
        m_list=[m0]; v_list=[v0]; u_seq=[];ui_seq=[]
        
        phiinv = torch.zeros_like(m0)
        phi = torch.zeros_like(m0)

        dt = 1/self.nsteps
        for i in range(self.nsteps):
            phiinv, m, v = self.EPDiff_step(m0, dt, phiinv)
            phi = phi + dt*lm.interp(v, phi)

            u_seq.append(phiinv)
            ui_seq.append(phi)

            if i<(self.nsteps-1):
                m_list.append(m); v_list.append(v)

        phiinv_disp_list = u_seq
        phi_disp_list = ui_seq
        # return u_seq, ui_seq, v_seq
        return phiinv_disp_list, phi_disp_list, v_list, m_list
    
    
    def IntWithVList(self, vseq):
        u_seq=[];ui_seq=[]
        phiinv = torch.zeros_like(vseq[0])
        phi = torch.zeros_like(vseq[0])

        dt = 1/self.nsteps
        
        for i in range(self.nsteps):
            v = vseq[i]
            phiinv = deform.compose_disp_vel(phiinv, v, dt=-dt)
            # phiinv = self.compose_disp_vel2(phiinv, v, dt=-dt)
            phi = phi + dt*lm.interp(v, phi)

            u_seq.append(phiinv)
            ui_seq.append(phi)

        phiinv_disp_list = u_seq
        phi_disp_list = ui_seq
        return phiinv_disp_list, phi_disp_list
    
    
    def compose2(self, u, v, ds=1.0, dt=1.0): #v， phiinv
        # #print("compose2!!!!!!!!!!!!!!!!!: ", u.shape, v.shape, ds, dt)  # u: [1, 2, 64, 64] v: [1, 2, 64, 64]
        # assert 4>345
        """Return ds*u(x) + dt*v(x + ds*u(x))"""
        return ds*u + dt*self.transformer(v, u, dt=ds)
    

    def compose_disp_vel2(self, u, v, dt=1.0):#phiinv, v
        """Given a displacement u, a velocity v, and a time step dt, compute
            dt*v(x) + u(x+dt*v(x))
        """
        return self.compose2(v, u, ds=dt, dt=1.0)
    
    def IntWithVList2(self, vseq):
        u_seq=[];ui_seq=[]
        phiinv = torch.zeros_like(vseq[0])
        phi = torch.zeros_like(vseq[0])

        dt = 1/self.nsteps
        
        for i in range(self.nsteps):
            # #print("!!!!!!!!!!!!!!!!!!!!!: ", vseq[i].shape, dt)  # phiinv: [1, 2, 64, 64] vseq[i]: [1, 2, 64, 64]
            v = vseq[i]
            # phiinv = self.transformer(phiinv, v, dt=-dt)  #wrong
            phiinv = self.compose_disp_vel2(phiinv, v, dt=-dt)
            phi = phi + dt*self.transformer(v, phi)

            u_seq.append(phiinv)
            ui_seq.append(phi)

        phiinv_disp_list = u_seq
        phi_disp_list = ui_seq

        return phiinv_disp_list, phi_disp_list
    
    
    # def ShootWithV02(self, v0):
    #     m0 = self.metric.flat(v0)
    #     m_list=[m0]; v_list=[v0]; u_seq=[];ui_seq=[]
        
    #     phiinv = torch.zeros_like(m0)
    #     phi = torch.zeros_like(m0)            

    #     dt = 1/self.nsteps
    #     for i in range(self.nsteps):
    #         phiinv, m, v = self.EPDiff_step(m0, dt, phiinv)
    #         phi = phi + dt*lm.interp(v, phi)

    #         u_seq.append(phiinv)
    #         ui_seq.append(phi)

    #         if i<(self.nsteps-1):
    #             m_list.append(m); v_list.append(v)

    #     phiinv_disp_list = u_seq
    #     phi_disp_list = ui_seq
    #     # return u_seq, ui_seq, v_seq




        return phiinv_disp_list, phi_disp_list, v_list, m_list
    

    

    





    def my_expmap_shooting_given_vseq(self, vseq, T=1.0, num_steps=10, phiinv=None, mommask=None, checkpoints=False):
        u_seq=[];ui_seq=[]

        if phiinv is None:
            phiinv = torch.zeros_like(vseq[0])
            phi = torch.zeros_like(vseq[0])

        if checkpoints is None or not checkpoints:
            dt = T/num_steps
            for i in range(num_steps):
                v = vseq[i]
                phiinv = deform.compose_disp_vel(phiinv, v, dt=-dt)
                u_seq.append(phiinv)
                phi = phi + dt*lm.interp(v, phi)
                ui_seq.append(phi)
        return phiinv, phi, u_seq, ui_seq
    
    def my_expmap_shooting(self, m0, T=1.0, num_steps=10, phiinv=None, mommask=None, checkpoints=False):
        """
        Given an initial momentum (Lie algebra element), compute the exponential
        map.

        What we return is actually only the inverse transformation phi^{-1}
        """
        # m_seq=[]; v_seq=[]; u_seq=[]
        d = len(m0.shape)-2

        if phiinv is None:
            phiinv = torch.zeros_like(m0)
            phi = torch.zeros_like(m0)

        if checkpoints is None or not checkpoints:
            # skip checkpointing
            dt = T/num_steps
            for i in range(num_steps):
                phiinv, m, v = self.EPDiff_step(m0, dt, phiinv, mommask=mommask)
                phi = phi + dt*lm.interp(v, phi)
        
        return phiinv, phi
    
    def my_get_u(self, v_seq=None, m_seq=None, T=1.0, num_steps=10, phiinv=None):
        if v_seq is None:
            if m_seq is None:
                assert 400>900
            v_seq = [self.metric.sharp(m) for m in m_seq]
        
        dt = T/num_steps
        if phiinv is None:
            phiinv = torch.zeros_like(v_seq[0])

        u_seq = [];phiinv_seq=[]
        for i in range(num_steps):
            phiinv = deform.compose_disp_vel(phiinv, v_seq[i], dt=-dt)
            u_seq.append(phiinv)
            # #print(torch.max(phiinv))
        # phiinv_seq = [u+deform.identity for u in u_seq]

        return u_seq
    
    




    def my_expmap_u2phi(self, m0, T=1.0, num_steps=10, phiinv=None, mommask=None, checkpoints=False):
        """
        Given an initial momentum (Lie algebra element), compute the exponential
        map.

        What we return is actually only the inverse transformation phi^{-1}
        """
        m_seq=[]; v_seq=[]; u_seq=[];ui_seq=[]
        d = len(m0.shape)-2
        v0 = self.metric.sharp(m0)
        m_seq.append(m0); v_seq.append(v0)

        if phiinv is None:
            phiinv = torch.zeros_like(m0)
            phi = torch.zeros_like(m0)

        if checkpoints is None or not checkpoints:
            dt = T/num_steps
            for i in range(num_steps):
                phiinv, m, v = self.EPDiff_step(m0, dt, phiinv, mommask=mommask)
                u_seq.append(phiinv)
                phi = phi + dt*lm.interp(v, phi)
                ui_seq.append(phi)

                if i<(num_steps-1):
                    m_seq.append(m); v_seq.append(v)

        return u_seq, ui_seq, v_seq
    
    def my_expmap(self, m0, T=1.0, num_steps=10, phiinv=None, mommask=None, checkpoints=False):
        # t1 = default_timer()
        """
        Given an initial momentum (Lie algebra element), compute the exponential
        map.

        What we return is actually only the inverse transformation phi^{-1}
        """
        m_seq=[]; v_seq=[]; u_seq=[]
        d = len(m0.shape)-2
        v0 = self.metric.sharp(m0)
        m_seq.append(m0); v_seq.append(v0)

        if phiinv is None:
            phiinv = torch.zeros_like(m0)

        if checkpoints is None or not checkpoints:
            # skip checkpointing
            dt = T/num_steps
            for i in range(num_steps):
                phiinv, m, v = self.EPDiff_step(m0, dt, phiinv, mommask=mommask)
                u_seq.append(phiinv)
                if i<(num_steps-1):
                    m_seq.append(m); v_seq.append(v)
        # #print("my_expmap: {}".format(default_timer()-t1))
        return u_seq,v_seq,m_seq
    
    def lagomorph_expmap_shootin(self, m0, T=1.0, num_steps=10, phiinv=None, mommask=None, checkpoints=False):
        """
        Given an initial momentum (Lie algebra element), compute the exponential map.

        What we return is actually only the inverse transformation phi^{-1}
        """
        d = len(m0.shape)-2

        if phiinv is None:
            phiinv = torch.zeros_like(m0)

        if checkpoints is None or not checkpoints:
            # skip checkpointing
            dt = T/num_steps
            for i in range(num_steps):
                phiinv, m, v = self.EPDiff_step(self.metric, m0, dt, phiinv, mommask=mommask)
                
        return phiinv
    
    def my_get_u2phi(self, v_seq=None, m_seq=None, T=1.0, num_steps=10, phiinv=None):
        # t1 = default_timer()
        if v_seq is None:
            if m_seq is None:
                assert 400>900
            v_seq = [self.metric.sharp(m) for m in m_seq]
        
        dt = T/num_steps
        if phiinv is None:
            phiinv = torch.zeros_like(v_seq[0])
            phi = torch.zeros_like(v_seq[0])

        u_seq = [];phiinv_seq=[];
        ui_seq = [];phi_seq=[]
        for i in range(num_steps):
            phiinv = deform.compose_disp_vel(phiinv, v_seq[i], dt=-dt)
            u_seq.append(phiinv)
            # #print(torch.max(phiinv))
            # phiinv_seq = [u+deform.identity(1, 2, 32,32) for u in u_seq]

            phi = phi + dt*lm.interp(v_seq[i], phi)
            ui_seq.append(phi)
            


        return u_seq, ui_seq

    def my_expmap_advect(self, m, T=1.0, num_steps=10, phiinv=None):
        """Compute EPDiff with vector momenta without using the integrated form.

        This is Euler integration of the following ODE:
            d/dt m = - ad_v^* m
        """
        v_seq = []; m_seq=[]
        d = len(m.shape)-2
        v0 = self.metric.sharp(m)
        m_seq.append(m); v_seq.append(v0)


        if phiinv is None:
            phiinv = torch.zeros_like(m)
        dt = T/num_steps
        v = self.metric.sharp(m)
        phiinv = deform.compose_disp_vel(phiinv, v, dt=-dt)
        v_seq.append(v); m_seq.append(m)


        for i in range(num_steps-1):
            m = m - dt*adjrep.ad_star(v, m)
            v = self.metric.sharp(m)
            phiinv = deform.compose_disp_vel(phiinv, v, dt=-dt)
            if i<(num_steps-2):
                v_seq.append(v); m_seq.append(m)
        return phiinv,v_seq,m_seq

class VecInt(nn.Module):
    def __init__(self, inshape, TSteps=7):
        super().__init__()
        self.nsteps = TSteps
        assert self.nsteps >= 0, 'nsteps should be >= 0, found: %d' % self.nsteps
        self.scale = 1.0 / (2 ** self.nsteps)

        # #print(self.nsteps)
        # assert 2>129

        self.transformer = SpatialTransformer(inshape)

    def forward(self, pos_flow):  #pos_flow: [b, 2, 64, 64]  (b,64,64,2)
        dims = len(pos_flow.shape)-2
        if dims == 2:
            b,c,w,h = pos_flow.shape
            if c != 2 and c != 3:
                pos_flow = pos_flow.permute(0,3,1,2)
        elif dims == 3:
            b,c,w,h,d = pos_flow.shape
            if c != 3:
                pos_flow = pos_flow.permute(0,4,1,2,3)

        vec = pos_flow
        dispList = []
        
        vec = vec * self.scale
        dispList.append(vec)

        for _ in range(self.nsteps):
            scratch,_ = self.transformer(vec, vec)
            vec = vec + scratch
            dispList.append(vec)

        scratch = vec
        for _ in range(self.nsteps):
            scratch,_ = self.transformer(scratch, scratch)


        phiinv_dispList = dispList
        return phiinv_dispList






class Hausdorff_distance:
    def __init__(self, percentile=95):
        self.kernel = self.get_kernel()
        self.percentile = percentile
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_kernel(self):
        kernel = np.array([[-1, -1, -1], [-1, 7.5, -1], [-1, -1, -1]], dtype=np.float32)
        return kernel

    def get_edge(self, seg):
        # #print(seg.shape, type(seg), seg.dtype, seg.max(), seg.min())
        outputs = cv2.filter2D(seg, -1, self.kernel)
        outputs = np.sign(outputs)
        return outputs
    
    def compute(self, array_a, array_b):
        edge_a = self.get_edge(array_a)
        edge_b = self.get_edge(array_b)

        position_a = np.where(edge_a == 1)
        position_b = np.where(edge_b == 1)

        xyz_a = np.array([position_a[0], position_a[1], position_a[2]]).T
        xyz_b = np.array([position_b[0], position_b[1], position_b[2]]).T

        distances_a_to_b  = torch.cdist(torch.tensor(xyz_a, dtype=torch.float32).cuda(), torch.tensor(xyz_b, dtype=torch.float32).cuda())
        distances_b_to_a  = torch.cdist(torch.tensor(xyz_b, dtype=torch.float32).cuda(), torch.tensor(xyz_a, dtype=torch.float32).cuda())

        min_a_to_b = torch.min(distances_a_to_b, dim=1).values
        min_b_to_a = torch.min(distances_b_to_a, dim=1).values


        # Compute classical or percentile Hausdorff distance
        hd_a_to_b = torch.quantile(min_a_to_b, self.percentile / 100.0)
        hd_b_to_a = torch.quantile(min_b_to_a, self.percentile / 100.0)

        distance = torch.max(torch.max(hd_a_to_b), torch.max(hd_b_to_a))

        return distance
    


def clear_labels(labels, type="oasis1"):
    maskL = masks[type]["maskL"]
    maskR = masks[type]["maskR"]
    maskT = masks[type]["maskT"]
    transf_label = labels.copy()
    # 处理 maskT
    mask = np.isin(transf_label, np.array(maskT))
    transf_label[~mask] = 0
    # 遍历 maskL 和 maskR
    for idx in range(len(maskL)):
        specific_values = np.array([maskL[idx], maskR[idx]])
        # 处理 transf_label
        mask = np.isin(transf_label, specific_values)
        transf_label[mask] = idx+1
    return transf_label



def jacobian_determinant(disp, grid=None, inshape=None):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    if grid is None:
        # compute grid
        # grid_lst = nd.volsize2ndgrid(volshape)
        # grid = np.stack(grid_lst, len(volshape))

        transformer = SpatialTransformer(**{
            'inshape': inshape,
            'mode': 'bilinear'
        })

        if nb_dims == 3:
            grid = transformer.grid.permute(0, 2, 3, 4, 1).cpu().numpy()[0]
        else:
            grid = transformer.grid.permute(0, 2, 3, 1).cpu().numpy()[0]


    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
    

def jacobian_det_for_all(outputs, ongpuGrad=False, TSteps=10, dir="", inshape=None, regW=2.0,  phiinv=True, templateIDX = -1):
    if phiinv:
        if templateIDX != -1:
            file = open(f"{dir}/result_{regW}_det_phiinv_T-{templateIDX}.txt", "a+")
        else:
            file = open(f"{dir}/result_{regW}_det_phiinv.txt", "a+")
    else:
        if templateIDX != -1:
            file = open(f"{dir}/result_{regW}_det_phi_T-{templateIDX}.txt", "a+")
        else:
            file = open(f"{dir}/result_{regW}_det_phi.txt", "a+")

    #print(file.name)
    if ongpuGrad: 
        disp = torch.cat([output["phiinv"] for output in outputs]).detach().cpu().numpy()    #(100, 2, 128, 128, 128)
        # phiinv_disp_list_9 = torch.cat([output["phiinv_disp_list"][9] for output in outputs]).detach().cpu().numpy()
    else:
        disp = torch.cat([output["phiinv"] for output in outputs]).cpu().numpy()
        # phiinv_disp_list_9 = torch.cat([output["phiinv_disp_list"][9] for output in outputs]).cpu().numpy()

    if len(disp.shape) == 5: #3D   1, 3, 128, 128, 128
        disp = np.transpose(disp, (0, 2, 3, 4, 1))  #100, 128, 128, 128, 3
    elif len(disp.shape) == 4: #2D
        disp = np.transpose(disp, (0, 2, 3, 1))  #100, 128, 128, 3
    
    # file.write(f"10_Steps{TSteps}_{disp.shape[0]}_LDDMM: \n")
    for idx in range(disp.shape[0]):
        det = jacobian_determinant(disp[idx], inshape=inshape)
        neg = np.sum(det<0) / det.size
        result_string_det = f'{neg}\n'
        file.write(result_string_det)



def jacobian_det_for_one(disp, inshape=None, exclude_boundary=1):
    # if ongpuGrad:
    #     disp = disp.detach().cpu().numpy()
    # else:
    #     disp = disp.cpu().numpy()

    if len(disp.shape) == 5: #3D   1, 3, 128, 128, 128
        disp = np.transpose(disp, (0, 2, 3, 4, 1))  #100, 128, 128, 128, 3
    elif len(disp.shape) == 4: #2D
        disp = np.transpose(disp, (0, 2, 3, 1))  #100, 128, 128, 3

    # print(disp.shape, inshape)  #(1, 64, 64, 2) (64, 64)
    # assert 3>129
    det = jacobian_determinant(disp[0], inshape=inshape)
    # print(det.shape) #(64, 64)

    # Exclude boundary pixels where gradient computation is less accurate
    if exclude_boundary > 0:
        det = det[exclude_boundary:-exclude_boundary, exclude_boundary:-exclude_boundary]

    neg = np.sum(det<0) / det.size

    return neg, det
        






def get_jacobian_det_for_all(outputs, ongpuGrad=False, inshape=None):
    if ongpuGrad: 
        disp = torch.cat([output["phiinv"] for output in outputs]).detach().cpu().numpy()    #(100, 2, 128, 128, 128)
        # phiinv_disp_list_9 = torch.cat([output["phiinv_disp_list"][9] for output in outputs]).detach().cpu().numpy()
    else:
        disp = torch.cat([output["phiinv"] for output in outputs]).cpu().numpy()
        # phiinv_disp_list_9 = torch.cat([output["phiinv_disp_list"][9] for output in outputs]).cpu().numpy()

    if len(disp.shape) == 5: #3D   1, 3, 128, 128, 128
        disp = np.transpose(disp, (0, 2, 3, 4, 1))  #100, 128, 128, 128, 3
    elif len(disp.shape) == 4: #2D
        disp = np.transpose(disp, (0, 2, 3, 1))  #100, 128, 128, 3
    
    dets = []
    #print(disp.shape)  #(10, 64, 64, 2)  cardiac

    for idx in range(disp.shape[0]):
        det = jacobian_determinant(disp[idx], inshape=inshape)
        #print(idx, "   ", det.shape)
        dets.append(det)
    
    # assert 3>129

    #expend a dimension
    dets = np.expand_dims(dets, axis=1)  #100, 128, 128, 128
    return np.stack(dets)  #100, 1, 128, 128, 128




# 定义 masks 字典
masks = {
    #OASIS1
    #1-Cerebral-White-Matter  0
    #6-Cerebellum-Cortex  4
    #2-Cerebral-Cortex 1
    # --------------------------------------
    #3-Lateral-Ventricle  2
    #7-Thalamus  5
    #8-Caudate   6
    #14-Hippocampus  8
    #9-Putamen   7
    # --------------------------------------
    #5-Cerebellum-White-Matter 3
    #13-Stem  9



    #1-Cerebral-White-Matter  0
    #2-Cerebral-Cortex 1
    #3-Lateral-Ventricle  2
    #5-Cerebellum-White-Matter 3
    #6-Cerebellum-Cortex  4
    #7-Thalamus  5
    #8-Caudate   6
    #9-Putamen   7
    #14-Hippocampus  8
    #13-Stem  9
    "oasis1": {
        "maskL": [1, 2, 3, 5, 6, 7, 8, 9, 14, 13],
        "maskR": [20, 21, 22, 24, 25, 26, 27, 28, 30, 13],
        "maskT": [1, 2, 3, 5, 6, 7, 8, 9, 14, 13, 20, 21, 22, 24, 25, 26, 27, 28, 30],
    },


    # "oasis1": {
    #     "maskL": [1, 2, 3, 5, 6, 7, 8, 9, 14, 13],
    #     "maskR": [20, 21, 22, 24, 25, 26, 27, 28, 30, 13],
    #     "maskT": [1, 2, 3, 5, 6, 7, 8, 9, 14, 13, 20, 21, 22, 24, 25, 26, 27, 28, 30],
    # },
    #OASIS3
    #2-Cerebral-White-Matter  0
    #3-Cerebral-Cortex 1
    #4-Lateral-Ventricle 2
    #7-Cerebellum-White-Matter 3
    #8-Cerebellum-Cortex 4
    #10-Thalamus 5
    #11-Caudate 6
    #12-Putamen  7
    #17-Hippocampus 8
    #16-Stem  9
    "oasis3": {
        "maskL": [2, 3, 4, 7, 8, 10, 11, 12, 17, 16],
        "maskR": [41, 42, 43, 46, 47, 49, 50, 51, 53, 16],
        "maskT": [2, 3, 4, 7, 8, 10, 11, 12, 17, 16, 41, 42, 43, 46, 47, 49, 50, 51, 53],
    }
}


hdorff_coefficient = Hausdorff_distance().compute
# 定义 Dice 系数计算函数
def dice_coefficient(array_a, array_b):
    intersection = np.sum(array_a * array_b)  # 元素相乘后求和
    union = np.sum(array_a) + np.sum(array_b)
    if union == 0:
        return 1.0  # 如果两者都为 0，返回 1.0
    return (2.0 * intersection) / union






def hausdorff_distance_95(tensor_a, tensor_b, spacing=None, return_mean=True, threshold=0.5):
    """
    Compute the 95% Hausdorff distance (HD95) between binary masks in tensor_a and tensor_b.
    Drops the largest 5% of point-wise distances (robust HD).

    Parameters
    ----------
    tensor_a, tensor_b : torch.Tensor
        Binary or probabilistic masks.
        Supported shapes:
          - [N, 1, H, W]
          - [N, 1, D, H, W]
          - [H, W] or [D, H, W] (no batch -> treated as batch size 1)
    spacing : sequence of floats or None
        Voxel spacing for each spatial dimension, e.g. (sx, sy, sz).
        If None, isotropic spacing of 1 is assumed.
    return_mean : bool
        If True, return mean HD95 over the batch (float).
        If False, return list of per-sample HD95 values.
    threshold : float
        Threshold to binarize masks (value > threshold is foreground).

    Returns
    -------
    float or list of floats
        Mean HD95 (if return_mean) or list of HD95 per sample.
    """

    # Clone to avoid modifying input
    tensor_a = tensor_a.clone()
    tensor_b = tensor_b.clone()

    # Make sure we have a batch dimension
    if tensor_a.ndim <= 3:   # e.g. [H, W] or [D, H, W]
        tensor_a = tensor_a.unsqueeze(0)
        tensor_b = tensor_b.unsqueeze(0)

    assert tensor_a.shape == tensor_b.shape, "tensor_a and tensor_b must have the same shape"

    # Shapes: [N, C, H, W] or [N, C, D, H, W]
    N = tensor_a.shape[0]
    spatial_shape = tensor_a.shape[2:]
    spatial_ndim = len(spatial_shape)

    if spacing is None:
        spacing = [1.0] * spatial_ndim
    spacing = torch.tensor(spacing, dtype=torch.float32, device=tensor_a.device)

    hd95_list = []

    for i in range(N):
        # collapse channel dim (index 1) to single mask by OR
        # (assumes shape [N, 1, ...], but works even if C>1)
        mask_a = (tensor_a[i] > threshold).any(dim=0)  # shape: spatial
        mask_b = (tensor_b[i] > threshold).any(dim=0)

        # Get coordinates of foreground voxels
        coords_a = mask_a.nonzero(as_tuple=False).float()  # [Na, spatial_ndim]
        coords_b = mask_b.nonzero(as_tuple=False).float()  # [Nb, spatial_ndim]

        # Handle empty masks
        if coords_a.numel() == 0 and coords_b.numel() == 0:
            # both empty -> distance 0
            hd95_list.append(0.0)
            continue
        elif coords_a.numel() == 0 or coords_b.numel() == 0:
            # one empty, one non-empty: conventionally HD is undefined or very large
            # you can choose your own convention; here we use NaN
            hd95_list.append(float('nan'))
            continue

        # Apply spacing to coordinates to work in physical space
        coords_a = coords_a * spacing
        coords_b = coords_b * spacing

        # Pairwise distances
        # dist_ab: [Na, Nb]
        dist_ab = torch.cdist(coords_a, coords_b, p=2)

        # Directed distances: for each point in A, minimum distance to B
        d_ab, _ = dist_ab.min(dim=1)  # [Na]
        d_ba, _ = dist_ab.min(dim=0)  # [Nb]

        # 95th percentile (drop largest 5%)
        # If there is only 1 point, quantile(0.95) is just that distance.
        q_ab = torch.quantile(d_ab, 0.95)
        q_ba = torch.quantile(d_ba, 0.95)

        # Symmetric HD95: max of the two directed 95% distances
        hd95 = torch.max(q_ab, q_ba)
        hd95_list.append(hd95.item())

    if return_mean:
        # ignore NaNs when averaging (in case of empty vs non-empty masks)
        vals = torch.tensor(hd95_list, dtype=torch.float32)
        mean_val = torch.nanmean(vals)
        return mean_val.item()
    else:
        return hd95_list



def dice_coefficient_tensor(tensor_a, tensor_b, return_mean=True):
    # print("dice_tensor_shape:  ",tensor_a.shape, tensor_b.shape)    #[10, 1, 128, 128] [10, 1, 128, 128]
    tensor_a = tensor_a.clone()
    tensor_b = tensor_b.clone()
    # print("dice_coefficient_tensor: ",tensor_a.shape, tensor_b.shape, "\n")

    # tensor_a[tensor_a<0.5] = 0
    # tensor_a[tensor_a>=0.5] = 1
    ndim = len(tensor_a.shape)
   
    if ndim == 3 or ndim == 2:
        intersection = torch.sum(tensor_a * tensor_b)
        union = torch.sum(tensor_a) + torch.sum(tensor_b)
    elif ndim == 4:
        intersection = torch.sum(tensor_a * tensor_b, dim=(1,2,3))  # Element-wise multiplication
        union = torch.sum(tensor_a, dim=(1,2,3)) + torch.sum(tensor_b,dim=(1,2,3))
    elif ndim == 5:
        intersection = torch.sum(tensor_a * tensor_b, dim=(1,2,3,4))  # Element-wise multiplication
        union = torch.sum(tensor_a, dim=(1,2,3,4)) + torch.sum(tensor_b,dim=(1,2,3,4))
    else:
        assert 4>9, "ndim is not 4 or 5 or 3 or 2"
    
    # print(intersection)
    # print(union)
    # if union == 0:
    #     return 1.0  # Handle the case where both tensors are empty.
    union[union==0] = 0.00000000001
    res = (2.0 * intersection) / union

    if return_mean:
        return torch.mean(res).item()
    res = [item.item() for item in res]
    return res




def smooth_mask_sdf(mask, sigma=1.0):
    mask = mask.astype(bool)

    dist_in  = distance_transform_edt(mask)
    dist_out = distance_transform_edt(~mask)
    sdf = dist_in - dist_out

    sdf_smooth = gaussian_filter(sdf, sigma=sigma)

    return (sdf_smooth > 0).astype(np.float32)



def smooth_mask_sdf_torch(mask, sigma=1.0):
    """
    mask: torch.Tensor [B, T, H, W], binary (0/1)
    return: torch.Tensor [B, T, H, W], binary (0/1)
    """
    device = mask.device
    mask_np = mask.detach().cpu().numpy().astype(bool)

    B, T, H, W = mask_np.shape
    out = np.zeros_like(mask_np, dtype=np.float32)

    for b in range(B):
        for t in range(T):
            m = mask_np[b, t]

            dist_in  = distance_transform_edt(m)
            dist_out = distance_transform_edt(~m)
            sdf = dist_in - dist_out

            sdf_smooth = gaussian_filter(sdf, sigma=sigma)
            out[b, t] = (sdf_smooth > 0).astype(np.float32)

    return torch.from_numpy(out).to(device)





def compute_dice_det(slices_series, Sdef_series, u_series, ui_series, Sdef_mask_series, lv_seg, batch_idx, inshape):  #input: [4, 9, 64, 64]
    # u_series.append(u)   #phiinv 对应的disp
    # ui_series.append(ui) #phi 对应的disp
    # slices_series, Sdef_series, v_series, u_series, ui_series, Sdef_mask_series, lv_seg, batch_idx

    # print(slices_series.shape, len(Sdef_series), Sdef_series[0].shape)
    #torch.Size([20, 15, 64, 64]) 14 torch.Size([20, 1, 64, 64])




    num_of_data = slices_series.shape[0]
    dice_list = []; det_list = []; hd95_list = []; sim_list = []; sim_list_masked = []; sim_list_masked_2=[]
    det_matrix_list = []
    for idx_o_d in range(num_of_data): #batch_size
        print(f"Processing {batch_idx}_{idx_o_d}th total length of data: {num_of_data}")
        # Processing 0_0th data: 24
        diff_list_inner=[]; dice_inner=[];hausdorff_inner=[];det_inner=[]; hd95_inner = []
        diff_sim_dinner_masked = []; diff_sim_dinner_masked_2 = []
        diff_sim_dinner=[];
        det_matrix_inner = []

        for frame in range(len(Sdef_series)):   #15个
            # print("idx_o_d  ", idx_o_d, "frame  ", frame, "\n\n")
            # Sdef_ = Sdef_series[frame][idx_o_d].detach().cpu().numpy()   #1,64,64
            # tar_ = slices_series[idx_o_d, frame+1:frame+2, :,:].cpu().numpy()  #1,64,64
            phi = ui_series[frame][idx_o_d:idx_o_d+1].detach().cpu().numpy()   #[64, 64, 2]
            neg_percent, det = jacobian_det_for_one(phi, inshape=inshape)
            det_inner.append(neg_percent)
            det_matrix_inner.append(det)


            sdef = Sdef_series[frame][idx_o_d].detach().cpu().numpy()
            tar = slices_series[idx_o_d, frame+1:frame+2,:,:].detach().cpu().numpy()
            diff_sim_dinner.append(np.mean(np.abs(sdef-tar)))
            



            # print(sdef.shape, tar.shape)
            # print("\n\n")
            # assert 4>109



            # continue
            if Sdef_mask_series is not None:
                Sdef_mask = Sdef_mask_series[frame][idx_o_d]   #1,64,64
                tar_mask = lv_seg[idx_o_d, frame+1:frame+2,:,:]  #1,64,64
                dice_one = dice_coefficient_tensor(Sdef_mask, tar_mask)
                hd95_one = hausdorff_distance_95(Sdef_mask, tar_mask)
                # print("-------------------tar_mask-------------------:  ", idx_o_d, frame, torch.sum(tar_mask))

                diff_sim_dinner_masked.append(np.mean(np.abs((sdef-tar)*tar_mask.detach().cpu().numpy())))
                diff_sim_dinner_masked_2.append(np.mean(np.abs((sdef-tar)*(1-tar_mask).detach().cpu().numpy())))




            else:
                dice_one = 0
                assert 4>4879
                
            dice_inner.append(dice_one)
            hd95_inner.append(hd95_one)
        dice_list.append(dice_inner)
        det_list.append(det_inner)
        hd95_list.append(hd95_inner)
        sim_list.append(diff_sim_dinner)
        sim_list_masked.append(diff_sim_dinner_masked)
        sim_list_masked_2.append(diff_sim_dinner_masked_2)
        det_matrix_list.append(det_matrix_inner)


    return dice_list, det_list, hd95_list, sim_list, det_matrix_list, sim_list_masked, sim_list_masked_2





def dice_coefficient_for_brain_one_example(dfm_label_fno, tarlabel, type="oasis1", hdorff=False):
    dice_one = []
    maskL = masks[type]["maskL"]
    maskR = masks[type]["maskR"]
    maskT = masks[type]["maskT"]

    # 遍历 maskL 和 maskR
    for idx in range(len(maskL)):
        specific_values = np.array([maskL[idx], maskR[idx]])

        # 处理 transf_label
        transf_label = dfm_label_fno.copy()
        mask = np.isin(transf_label, specific_values)
        transf_label[mask] = 1
        transf_label[~mask] = 0

        # 处理 target_label
        target_label = tarlabel.copy()
        mask = np.isin(target_label, specific_values)
        target_label[mask] = 1
        target_label[~mask] = 0

        # 计算 Dice 系数
        dice_transf = dice_coefficient(transf_label, target_label)
        dice_one.append(dice_transf); 


    # 处理 maskT
    transf_label = dfm_label_fno.copy()
    mask = np.isin(transf_label, np.array(maskT))
    transf_label[mask] = 1
    transf_label[~mask] = 0

    target_label = tarlabel.copy()
    mask = np.isin(target_label, np.array(maskT))
    target_label[mask] = 1
    target_label[~mask] = 0

    # 计算 Dice 系数
    dice_transf = dice_coefficient(transf_label, target_label)
    dice_one.append(dice_transf)

    return dice_one, None



def hdorff_coefficient_for_brain_one_example(dfm_label_fno, tarlabel, type="oasis1"):
    hdorff_one = []
    maskL = masks[type]["maskL"]
    maskR = masks[type]["maskR"]
    maskT = masks[type]["maskT"]

    # 遍历 maskL 和 maskR
    for idx in range(len(maskL)):
        specific_values = np.array([maskL[idx], maskR[idx]])

        # 处理 transf_label
        transf_label = dfm_label_fno.copy()
        mask = np.isin(transf_label, specific_values)
        transf_label[mask] = 1
        transf_label[~mask] = 0

        # 处理 target_label
        target_label = tarlabel.copy()
        mask = np.isin(target_label, specific_values)
        target_label[mask] = 1
        target_label[~mask] = 0

        # 计算 Dice 系数
        hdorff_transf = hdorff_coefficient(transf_label, target_label)
        hdorff_one.append(hdorff_transf.item())


    """ # 处理 maskT
    transf_label = dfm_label_fno.copy()
    mask = np.isin(transf_label, np.array(maskT))
    transf_label[mask] = 1
    transf_label[~mask] = 0

    target_label = tarlabel.copy()
    mask = np.isin(target_label, np.array(maskT))
    target_label[mask] = 1
    target_label[~mask] = 0
    hdorff_transf = hdorff_coefficient(transf_label, target_label)
    hdorff_one.append(hdorff_transf.item()) """

    return hdorff_one



def dice_coefficient_for_cardiac(outputs, ongpuGrad=False, TSteps=10, dir="", type="oasis1", regW=2.0, hdorff=False, templateIDX=-1):
    if templateIDX != -1:
        file = open(f"{dir}/result_{regW}_dice_T-{templateIDX}.txt", "a+")
    else:
        file = open(f"{dir}/result_{regW}_dice.txt", "a+")
    
    if ongpuGrad:
        tarlabels = torch.cat([output["tarlabel"] for output in outputs]).detach().cpu().numpy()    #(100, 1, 128, 128, 128)
        dfmSrclabels = torch.cat([output["dfmSrclabel"] for output in outputs]).detach().cpu().numpy()
    else:
        tarlabels = torch.cat([output["tarlabel"] for output in outputs]).cpu().numpy()    #(100, 1, 128, 128, 128)
        dfmSrclabels = torch.cat([output["dfmSrclabel"] for output in outputs]).cpu().numpy()  #(100, 1, 128, 128, 128)


    dice_all = []; hdorff_all = []
    # #print(dfmSrclabels.shape)  #(5, 1, 64, 64)


    for i in range(dfmSrclabels.shape[0]):
        dice_one = dice_coefficient(dfmSrclabels[i][0], tarlabels[i][0])
        dice_all.append(dice_one)
        str_dice = ' '.join(map(str, [dice_one]))
        file.write(str_dice+"\n")
        #print("Dice: ",  dice_one)
    file.close()
    return dice_all, []


def dice_coefficient_for_brain(outputs, ongpuGrad=False, TSteps=10, dir="", type="oasis1", regW=2.0, hdorff=False, templateIDX=-1):
    if templateIDX != -1:
        file = open(f"{dir}/result_{regW}_dice_T-{templateIDX}.txt", "a+")
    else:
        file = open(f"{dir}/result_{regW}_dice.txt", "a+")
    
    #print(file.name)

    if ongpuGrad:
        tarlabels = torch.cat([output["tarlabel"] for output in outputs]).detach().cpu().numpy()    #(100, 1, 128, 128, 128)
        dfmSrclabels = torch.cat([output["dfmSrclabel"] for output in outputs]).detach().cpu().numpy()
    else:
        tarlabels = torch.cat([output["tarlabel"] for output in outputs]).cpu().numpy()    #(100, 1, 128, 128, 128)
        dfmSrclabels = torch.cat([output["dfmSrclabel"] for output in outputs]).cpu().numpy()  #(100, 1, 128, 128, 128)

    file.write(f"10_Steps{TSteps}_{dfmSrclabels.shape[0]}_LDDMM: \n")

    dice_all = []; hdorff_all = []
    for i in range(dfmSrclabels.shape[0]):
        dice_one, hdorff_one = dice_coefficient_for_brain_one_example(dfmSrclabels[i][0], tarlabels[i][0], type=type, hdorff=hdorff)
        dice_all.append(dice_one)
        hdorff_all.append(hdorff_one)
        #print("Dice: ",  dice_one)
        file.write(' '.join(map(str, dice_one))+"\n")


import time
def hdorff_coefficient_for_brain(outputs, ongpuGrad=False, TSteps=10, dir="", type="oasis1", regW=2.0, templateIDX=-1):
    if templateIDX != -1:
        file_hdorff = open(f"{dir}/result_{regW}_hdorff_T-{templateIDX}.txt", "a+")
    else:
        file_hdorff = open(f"{dir}/result_{regW}_hdorff.txt", "a+")
    #print(file_hdorff.name)

    if ongpuGrad:
        tarlabels = torch.cat([output["tarlabel"] for output in outputs]).detach().cpu().numpy()    #(100, 1, 128, 128, 128)
        dfmSrclabels = torch.cat([output["dfmSrclabel"] for output in outputs]).detach().cpu().numpy()
    else:
        tarlabels = torch.cat([output["tarlabel"] for output in outputs]).cpu().numpy()    #(100, 1, 128, 128, 128)
        dfmSrclabels = torch.cat([output["dfmSrclabel"] for output in outputs]).cpu().numpy()  #(100, 1, 128, 128, 128)

    file_hdorff.write(f"10_Steps{TSteps}_{dfmSrclabels.shape[0]}_LDDMM: \n")
    hdorff_all = []
    for i in range(dfmSrclabels.shape[0]):
        time_start = time.time()
        #print("computing hdorff for {}th image".format(i))
        hdorff_one = hdorff_coefficient_for_brain_one_example(dfmSrclabels[i][0], tarlabels[i][0], type=type)
        hdorff_all.append(hdorff_one)
        #print("hdorff: ",  hdorff_one)
        file_hdorff.write(' '.join(map(str, hdorff_one))+"\n")
        time_end = time.time()
        #print("Time taken for {}th image: {:.2f} seconds".format(i, time_end - time_start))












from matplotlib import pyplot as plt
def to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    try:
        from pycuda import gpuarray
        if isinstance(arr, gpuarray.GPUArray):
            return arr.get()
    except ImportError:
        pass
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
    except ImportError:
        pass

    raise Exception(f"Cannot convert type {type(arr)} to numpy.ndarray.")

def Mgridplot(u, ax, Nx=64, Ny=64, displacement=True, linewidth=0.5, color='red', IMG=None, **kwargs):
    """Given a displacement field, plot a displaced grid"""
    u = to_numpy(u)  #(1, 2, 128, 128)

    

    assert u.shape[0] == 1, "Only send one deformation at a time"
    
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.axis('off')  # Remove axis
    ax.set_adjustable('box')  # Ensure the aspect ratio is consistent for each plot
    
    if IMG is not None:
        # ax.imshow(IMG, cmap='gray', alpha=0.5)
        ax.imshow(IMG, cmap='gray')

    if Nx is None:
        Nx = u.shape[2]
    if Ny is None:
        Ny = u.shape[3]
    # downsample displacements
    h = np.copy(u[0,:,::u.shape[2]//Nx, ::u.shape[3]//Ny])

    # now reset to actual Nx Ny that we achieved
    Nx = h.shape[1]
    Ny = h.shape[2]
    # adjust displacements for downsampling
    h[0,...] /= float(u.shape[2])/Nx
    h[1,...] /= float(u.shape[3])/Ny





    if displacement: # add identity
        '''
            h[0]: 
        '''
        h[0,...] += np.arange(Nx).reshape((Nx,1))  #h[0]:  (118, 109)  add element: 118*1
        h[1,...] += np.arange(Ny).reshape((1,Ny))

    # put back into original index space
    h[0,...] *= float(u.shape[2])/Nx
    h[1,...] *= float(u.shape[3])/Ny


    from scipy.ndimage import zoom
    upscale = 4
    h_smooth = zoom(h, (1, upscale, upscale), order=3)  # cubic interpolation
    h = np.copy(h_smooth)


    
    # create a meshgrid of locations
    for i in range(h.shape[1]):
        # ax.plot( h[0,i,:], h[1,i,:], color=color, linewidth=1.0, **kwargs)
        ax.plot(h[1, i, :], h[0, i, :], color=color, linewidth=linewidth, **kwargs)
    for i in range(h.shape[2]):
        # ax.plot(h[0,:,i], h[1,:,i],  color=color, linewidth=1.0, **kwargs)
        ax.plot(h[1, :, i],h[0, :, i] , color=color, linewidth=linewidth, **kwargs)

    # Adjust axis limits to remove unnecessary whitespace
    ax.set_xlim(h[1].min(), h[1].max())  
    ax.set_ylim(h[0].min(), h[0].max())  
    ax.margins(0)  
    ax.set_adjustable('datalim')
    ax.set_aspect('equal')
    ax.invert_yaxis()



    # ### rotate 90° clockwise
    # H = u.shape[2]  # height
    # W = u.shape[3]  # width
    # x = h[1]  # (Nx, Ny)
    # y = h[0]  # (Nx, Ny)
    # # rotate 90° clockwise: (x, y) -> (y, W-1-x)
    # x_rot = y
    # y_rot = (W - 1) - x
    # for i in range(x_rot.shape[0]):
    #     ax.plot(x_rot[i, :], y_rot[i, :], color=color, linewidth=linewidth, **kwargs)
    # for i in range(x_rot.shape[1]):
    #     ax.plot(x_rot[:, i], y_rot[:, i], color=color, linewidth=linewidth, **kwargs)
    # # limits
    # ax.set_xlim(x_rot.min(), x_rot.max())
    # ax.set_ylim(y_rot.min(), y_rot.max())
    # ax.margins(0)
    # ax.set_aspect('equal')
    




def MgridplotArr(
    u, ax, Nx=64, Ny=64, color="red", units="xy", angles="xy", scale=1.0, **kwargs
):
    """Given a displacement field, plot a quiver of vectors"""
    u = to_numpy(u)
    assert u.shape[0] == 1, "Only send one deformation at a time"
    assert u.ndim == 4, "Only 2D deformations can use quiver()"
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.axis('off')  # Remove axis
    ax.set_adjustable('box')  # Ensure the aspect ratio is consistent for each plot
    

    if Nx is None:
        Nx = u.shape[2]
    if Ny is None:
        Ny = u.shape[3]
    # downsample displacements
    h = np.copy(u[:, :, :: u.shape[2] // Nx, :: u.shape[3] // Ny])
    ix = lm.identity(u.shape, u.dtype)[:, :, :: u.shape[2] // Nx, :: u.shape[3] // Ny]
    # #print("ix: ", ix.shape) #ix:  (1, 2, 32, 32)
    # assert 3>222
    # create a meshgrid of locations
    # #print(h.shape) #(1, 2, 32, 32)
    # assert 3>222
    magnitude = np.sqrt(h[0, 1, :, :]**2 + h[0, 0, :, :]**2)
    ax.quiver(
        ix[0, 1, :, :],
        ix[0, 0, :, :],
        h[0, 1, :, :],
        h[0, 0, :, :],
        color=color,
        angles=angles,
        units=units,
        # scale=scale,

        scale=None,         # Auto-scale arrows
        width=1.0,         # 调整箭头线的粗细 
        headwidth=12,        # 增加箭头头部宽度  
        headlength=15,       # 增加箭头头部长度  
        # pivot="middle",      # Ensures only arrows are drawn (no dots at tails)
        # pivot="tip",      # Ensures only arrows are drawn (no dots at tails)
        # alpha=(magnitude > 0.5).astype(float)  # 透明度为 0 的部分不会显示
    )

    



def add_zero_channel_and_norm(VecF):
    # #print(VecF.dtype)
    # #print("max-", VecF.max(), "min-", VecF.min(), "mean-", VecF.mean())
    # VecF = (VecF - VecF.min()) / (VecF.max() - VecF.min())
    # #print("max-", VecF.max(), "min-", VecF.min(), "mean-", VecF.mean())
    
    
    if len(VecF.shape) == 4:
        zeros_channel = np.zeros((VecF.shape[0], 1, VecF.shape[2], VecF.shape[3]), dtype=VecF.dtype)
        VecF = np.concatenate((VecF, zeros_channel), axis=1)
        # #print(VecF.dtype)
        VecF = np.transpose(VecF, (0, 2, 3, 1))
    elif len(VecF.shape) == 5:
        VecF = np.transpose(VecF, (0, 2, 3, 4, 1))

    for i in range(VecF.shape[0]):
        img = VecF[i]  # Extract single image (32, 32)
        min_val, max_val = img.min(), img.max()
        VecF[i] = (img - min_val) / (max_val - min_val)
    

    return VecF


def plot_norm_list_gpu(outputs, regmetric, mse_metric, ongpuGrad=False, TSteps=10, dir="", saveRes=True):
    if not os.path.exists(dir) and dir != "":
        os.makedirs(dir)
    #print(dir)
    savedir = dir

    VecF_0 = torch.cat([output["VecF_List"][0] for output in outputs]) #torch.Size([100, 2, 32, 32])
    VecF_1 = torch.cat([output["VecF_List"][1] for output in outputs])
    VecF_2 = torch.cat([output["VecF_List"][2] for output in outputs])
    VecF_3 = torch.cat([output["VecF_List"][3] for output in outputs])  
    VecF_4 = torch.cat([output["VecF_List"][4] for output in outputs])

    
    



    if TSteps == 10:
        VecF_5 = torch.cat([output["VecF_List"][5] for output in outputs])
        VecF_6 = torch.cat([output["VecF_List"][6] for output in outputs])
        VecF_7 = torch.cat([output["VecF_List"][7] for output in outputs])
        VecF_8 = torch.cat([output["VecF_List"][8] for output in outputs])
        VecF_9 = torch.cat([output["VecF_List"][9] for output in outputs])
        VecF_List_Tensor = torch.stack([VecF_0, VecF_1, VecF_2, VecF_3, VecF_4, VecF_5, VecF_6, VecF_7, VecF_8, VecF_9], dim=0)
    else:
        VecF_List_Tensor = torch.stack([VecF_0, VecF_1, VecF_2, VecF_3, VecF_4], dim=0)

    if saveRes:
        if ongpuGrad:
            # # torch.Size([10, 10, 3, 128, 128, 128]) torch.Size([10, 1, 128, 128, 128])
            nums = VecF_List_Tensor[0].shape[0]
            for i in range(TSteps):
                np.save(f"{savedir}/10_Steps{TSteps}_{nums}_LDDMM_VecF_{i}.npy", VecF_List_Tensor[i].detach().cpu().numpy())
        else:
            #delete
            nums = VecF_List_Tensor[0].shape[0]
            for i in range(TSteps):
                np.save(f"{savedir}/10_Steps{TSteps}_{nums}_LDDMM_VecF_{i}.npy", VecF_List_Tensor[i].cpu().numpy())





    MomF_0 = torch.cat([output["MomF_List"][0] for output in outputs])
    MomF_1 = torch.cat([output["MomF_List"][1] for output in outputs])
    MomF_2 = torch.cat([output["MomF_List"][2] for output in outputs])
    MomF_3 = torch.cat([output["MomF_List"][3] for output in outputs])
    MomF_4 = torch.cat([output["MomF_List"][4] for output in outputs])
    if TSteps == 10:
        MomF_5 = torch.cat([output["MomF_List"][5] for output in outputs])
        MomF_6 = torch.cat([output["MomF_List"][6] for output in outputs])
        MomF_7 = torch.cat([output["MomF_List"][7] for output in outputs])
        MomF_8 = torch.cat([output["MomF_List"][8] for output in outputs])
        MomF_9 = torch.cat([output["MomF_List"][9] for output in outputs])
        MomF_List_Tensor = torch.stack([MomF_0, MomF_1, MomF_2, MomF_3, MomF_4, MomF_5, MomF_6, MomF_7, MomF_8, MomF_9], dim=0)
    else:
        MomF_List_Tensor = torch.stack([MomF_0, MomF_1, MomF_2, MomF_3, MomF_4], dim=0)

    
    # #print(VecF_List_Tensor.shape) #[10, 100, 2, 32, 32] [10, 100, 2, 32, 32]
    # assert 3>111
    reg_list_list = []; reg_mean_list = []

    for i in range(MomF_List_Tensor.shape[1]): #循环100次
        reg_list = regmetric.comput_loss_list(MomF_List_Tensor[:, i:i+1], VecF_List_Tensor[:, i:i+1])
        # #print(reg_list) #[0.7382804155349731, 0.7375214099884033, 1.1307733058929443, 3.7936384677886963, 10.891395568847656]
        # assert 3>129
        reg_list_list.append(reg_list)
        reg_mean_list.append(torch.tensor(sum(reg_list) / len(reg_list)))
        # #print(i, reg_list)
            # 转换数据为numpy数组以便处理
    data = np.array(reg_list_list)
    # 统计每个位置的数据分布，可以使用boxplot或者violinplot
    plt.figure(figsize=(TSteps, 6))
    # 绘制箱型图，每列代表每个位置的数据分布
    plt.boxplot(data, vert=True)  # vert=False 表示将箱型图绘制为水平
    # 设置标题和标签
    plt.title("Norm (v*m) of velocity over time")
    plt.xlabel(f"Time (1 to {TSteps})")
    plt.ylabel("values")

    # 显示图表
    plt.show()
    
    srcs = torch.cat([output["src"] for output in outputs])   #(100, 1, 32, 32)
    tars = torch.cat([output["tar"] for output in outputs])   #(100, 1, 32, 32)
    preds = torch.cat([output["pred"] for output in outputs]) #(100, 1, 32, 32)
    mse_list = []
    for i in range(tars.shape[0]): #循环100次
        mse_list.append(mse_metric(tars[i:i+1], preds[i:i+1]))

    
    #print("reg_list\n", reg_list)
    reg_list_string = ' '.join(map(str, reg_list))+"\n"
    file = open(f"{dir}/lddmm_plot_geodesic_reg.txt", "a+")
    file.write(f"{reg_list_string}\n")


    #print("mse_list: ", torch.stack(mse_list), "mean: ", torch.stack(mse_list).mean())
    #print("reg_mean_list: ", torch.stack(reg_mean_list), "mean: ", torch.stack(reg_mean_list).mean())


def save_dfm_dfmlabel(outputs, ongpuGrad=False, dir="", save_list = [0,2]):
    #print(dir)  #/home/nellie/code/cvpr/Project/DynamiCrafter/2025_SAVE_ROOT_DIR/2025training_oasisv2.3_simplenop
    if ongpuGrad: 
        srcs = torch.cat([output["src"] for output in outputs]).detach().cpu().numpy()
        tars = torch.cat([output["tar"] for output in outputs]).detach().cpu().numpy()
        srclabels = torch.cat([output["srclabel"] for output in outputs]).detach().cpu().numpy()
        tarlabels = torch.cat([output["tarlabel"] for output in outputs]).detach().cpu().numpy()
        dfmSrclabels = torch.cat([output["dfmSrclabel"] for output in outputs]).detach().cpu().numpy()
        preds = torch.cat([output["pred"] for output in outputs]).detach().cpu().numpy()
    else:
        srcs = torch.cat([output["src"] for output in outputs]).cpu().numpy()
        tars = torch.cat([output["tar"] for output in outputs]).cpu().numpy()
        srclabels = torch.cat([output["srclabel"] for output in outputs]).cpu().numpy()
        tarlabels = torch.cat([output["tarlabel"] for output in outputs]).cpu().numpy()
        dfmSrclabels = torch.cat([output["dfmSrclabel"] for output in outputs]).cpu().numpy()
        preds = torch.cat([output["pred"] for output in outputs]).cpu().numpy()

    # #print("\n\n srcs: ", srcs.shape, "tars: ", tars.shape, "srclabels: ", srclabels.shape, "tarlabels: ", tarlabels.shape, "dfmSrclabel: ", dfmSrclabels.shape)
    # srcs:  (10, 1, 128, 128, 128) tars:  (10, 1, 128, 128, 128) srclabels:  (10, 1, 128, 128, 128) tarlabels:  (10, 1, 128, 128, 128) dfmSrclabel:  (10, 1, 128, 128, 128)
    #save them as .nii.gz
    import SimpleITK as sitk
    for idx in save_list:
        src = srcs[idx, 0]
        tar = tars[idx, 0]
        srclabel = srclabels[idx, 0]
        tarlabel = tarlabels[idx, 0]
        dfmSrclabel = dfmSrclabels[idx, 0]
        pred = preds[idx, 0]

        src = sitk.GetImageFromArray(src)
        tar = sitk.GetImageFromArray(tar)
        srclabel = sitk.GetImageFromArray(srclabel)
        tarlabel = sitk.GetImageFromArray(tarlabel)
        dfmSrclabel = sitk.GetImageFromArray(dfmSrclabel)
        pred = sitk.GetImageFromArray(pred)

        sitk.WriteImage(src, f"{dir}/src_{idx}.nii.gz")
        sitk.WriteImage(tar, f"{dir}/tar_{idx}.nii.gz")
        sitk.WriteImage(srclabel, f"{dir}/srclabel_{idx}.nii.gz")
        sitk.WriteImage(tarlabel, f"{dir}/tarlabel_{idx}.nii.gz")
        sitk.WriteImage(dfmSrclabel, f"{dir}/dfmSrclabel_{idx}.nii.gz")
        sitk.WriteImage(pred, f"{dir}/pred_{idx}.nii.gz")


def collect_outputs(outputs,ongpuGrad=False,TSteps=10,dir="",mse_metric=None,saveRes=True,JabDet=None,vie=-1):
    savedir = dir
    # #print(len(outputs)) #1
    # for output in outputs:
    #     #print(output["src"].shape) #[100, 1, 32, 32]
    # assert 3>123

    if ongpuGrad:  #LDDMM Optimize
        srcs = torch.cat([output["src"] for output in outputs]).detach().cpu().numpy()
        tars = torch.cat([output["tar"] for output in outputs]).detach().cpu().numpy()
        preds = torch.cat([output["pred"] for output in outputs]).detach().cpu().numpy()
        if len(srcs.shape) == 5: #3D
            dfmSrclabels = torch.cat([output["dfmSrclabel"] for output in outputs]).detach().cpu().numpy()
            srclabels = torch.cat([output["srclabel"] for output in outputs]).detach().cpu().numpy()
            tarlabels = torch.cat([output["tarlabel"] for output in outputs]).detach().cpu().numpy()
        elif len(srcs.shape) == 4: #2D
            dfmSrclabels = None
        #get the VecF_List
        VecF_0 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][0] for output in outputs]).detach().cpu().numpy())
        VecF_1 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][1] for output in outputs]).detach().cpu().numpy())
        VecF_2 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][2] for output in outputs]).detach().cpu().numpy())
        VecF_3 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][3] for output in outputs]).detach().cpu().numpy())
        VecF_4 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][4] for output in outputs]).detach().cpu().numpy())
        if TSteps != 5:
            VecF_5 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][5] for output in outputs]).detach().cpu().numpy())
            VecF_6 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][6] for output in outputs]).detach().cpu().numpy())
            VecF_7 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][7] for output in outputs]).detach().cpu().numpy())
            VecF_8 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][8] for output in outputs]).detach().cpu().numpy())
            VecF_9 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][9] for output in outputs]).detach().cpu().numpy())
        #phiinv_disp_list
        phiinv_disp_list_0 = torch.cat([output["phiinv_disp_list"][0] for output in outputs]).detach().cpu().numpy()
        phiinv_disp_list_1 = torch.cat([output["phiinv_disp_list"][1] for output in outputs]).detach().cpu().numpy()
        phiinv_disp_list_2 = torch.cat([output["phiinv_disp_list"][2] for output in outputs]).detach().cpu().numpy()
        phiinv_disp_list_3 = torch.cat([output["phiinv_disp_list"][3] for output in outputs]).detach().cpu().numpy()
        phiinv_disp_list_4 = torch.cat([output["phiinv_disp_list"][4] for output in outputs]).detach().cpu().numpy()
        if TSteps != 5:
            phiinv_disp_list_5 = torch.cat([output["phiinv_disp_list"][5] for output in outputs]).detach().cpu().numpy()
            phiinv_disp_list_6 = torch.cat([output["phiinv_disp_list"][6] for output in outputs]).detach().cpu().numpy()
            phiinv_disp_list_7 = torch.cat([output["phiinv_disp_list"][7] for output in outputs]).detach().cpu().numpy()
            phiinv_disp_list_8 = torch.cat([output["phiinv_disp_list"][8] for output in outputs]).detach().cpu().numpy()
            phiinv_disp_list_9 = torch.cat([output["phiinv_disp_list"][9] for output in outputs]).detach().cpu().numpy()
        #dfm list
        dfmSrc_list_0 = torch.cat([output["dfmSrc_list"][0] for output in outputs]).detach().cpu().numpy()
        dfmSrc_list_1 = torch.cat([output["dfmSrc_list"][1] for output in outputs]).detach().cpu().numpy()
        dfmSrc_list_2 = torch.cat([output["dfmSrc_list"][2] for output in outputs]).detach().cpu().numpy()
        dfmSrc_list_3 = torch.cat([output["dfmSrc_list"][3] for output in outputs]).detach().cpu().numpy()
        dfmSrc_list_4 = torch.cat([output["dfmSrc_list"][4] for output in outputs]).detach().cpu().numpy()
        if TSteps != 5:
            dfmSrc_list_5 = torch.cat([output["dfmSrc_list"][5] for output in outputs]).detach().cpu().numpy()
            dfmSrc_list_6 = torch.cat([output["dfmSrc_list"][6] for output in outputs]).detach().cpu().numpy()
            dfmSrc_list_7 = torch.cat([output["dfmSrc_list"][7] for output in outputs]).detach().cpu().numpy()
            dfmSrc_list_8 = torch.cat([output["dfmSrc_list"][8] for output in outputs]).detach().cpu().numpy()
            dfmSrc_list_9 = torch.cat([output["dfmSrc_list"][9] for output in outputs]).detach().cpu().numpy()


        #save the data for compute metrics like: DICE, Jacobian, VecF
        # VecF_list = [VecF_0, VecF_1, VecF_2, VecF_3, VecF_4, VecF_5, VecF_6, VecF_7, VecF_8, VecF_9]
        # phiinv_disp_list_list = [phiinv_disp_list_0, phiinv_disp_list_1, phiinv_disp_list_2, phiinv_disp_list_3, phiinv_disp_list_4, phiinv_disp_list_5, phiinv_disp_list_6, phiinv_disp_list_7, phiinv_disp_list_8, phiinv_disp_list_9]
        # dfmSrc_list_list = [dfmSrc_list_0, dfmSrc_list_1, dfmSrc_list_2, dfmSrc_list_3, dfmSrc_list_4, dfmSrc_list_5, dfmSrc_list_6, dfmSrc_list_7, dfmSrc_list_8, dfmSrc_list_9]
        if TSteps != 5:
            VecF_list = [VecF_0, VecF_1, VecF_2, VecF_3, VecF_4, VecF_5, VecF_6, VecF_7, VecF_8, VecF_9]
            phiinv_disp_list_list = [phiinv_disp_list_0, phiinv_disp_list_1, phiinv_disp_list_2, phiinv_disp_list_3, phiinv_disp_list_4, phiinv_disp_list_5, phiinv_disp_list_6, phiinv_disp_list_7, phiinv_disp_list_8, phiinv_disp_list_9]
            dfmSrc_list_list = [dfmSrc_list_0, dfmSrc_list_1, dfmSrc_list_2, dfmSrc_list_3, dfmSrc_list_4, dfmSrc_list_5, dfmSrc_list_6, dfmSrc_list_7, dfmSrc_list_8, dfmSrc_list_9]
        else:
            VecF_list = [VecF_0, VecF_1, VecF_2, VecF_3, VecF_4]
            phiinv_disp_list_list = [phiinv_disp_list_0, phiinv_disp_list_1, phiinv_disp_list_2, phiinv_disp_list_3, phiinv_disp_list_4]
            dfmSrc_list_list = [dfmSrc_list_0, dfmSrc_list_1, dfmSrc_list_2, dfmSrc_list_3, dfmSrc_list_4]

        if saveRes:
            nums = srcs.shape[0]
            np.save(f"{savedir}/10_Steps{TSteps}_{nums}_LDDMM_dfmSrclabels.npy", dfmSrclabels)
            for i in range(TSteps): 
                np.save(f"{savedir}/10_Steps{TSteps}_{nums}_LDDMM_dfmSrc_list_{i}.npy", dfmSrc_list_list[i])
                np.save(f"{savedir}/10_Steps{TSteps}_{nums}_LDDMM_phiinv_disp_list_{i}.npy", phiinv_disp_list_list[i])

    else:
        srcs = torch.cat([output["src"] for output in outputs]).cpu().numpy()
        tars = torch.cat([output["tar"] for output in outputs]).cpu().numpy()
        preds = torch.cat([output["pred"] for output in outputs]).cpu().numpy()

        if len(srcs.shape) == 5: #3D
            srclabels = torch.cat([output["srclabel"] for output in outputs]).cpu().numpy()
            dfmSrclabels = torch.cat([output["dfmSrclabel"] for output in outputs]).cpu().numpy()  #(100, 1, 128, 128, 128)
            tarlabels = torch.cat([output["tarlabel"] for output in outputs]).cpu().numpy()
        elif len(srcs.shape) == 4: #2D
            dfmSrclabels = None

        #get the VecF_List
        VecF_0 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][0] for output in outputs]).cpu().numpy())
        VecF_1 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][1] for output in outputs]).cpu().numpy())
        VecF_2 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][2] for output in outputs]).cpu().numpy())
        VecF_3 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][3] for output in outputs]).cpu().numpy())
        VecF_4 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][4] for output in outputs]).cpu().numpy())
        if TSteps != 5:
            VecF_5 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][5] for output in outputs]).cpu().numpy())
            VecF_6 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][6] for output in outputs]).cpu().numpy())
            VecF_7 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][7] for output in outputs]).cpu().numpy())
            VecF_8 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][8] for output in outputs]).cpu().numpy())
            VecF_9 = add_zero_channel_and_norm(torch.cat([output["VecF_List"][9] for output in outputs]).cpu().numpy())
        #phiinv_disp_list
        phiinv_disp_list_0 = torch.cat([output["phiinv_disp_list"][0] for output in outputs]).cpu().numpy()
        phiinv_disp_list_1 = torch.cat([output["phiinv_disp_list"][1] for output in outputs]).cpu().numpy()
        phiinv_disp_list_2 = torch.cat([output["phiinv_disp_list"][2] for output in outputs]).cpu().numpy()
        phiinv_disp_list_3 = torch.cat([output["phiinv_disp_list"][3] for output in outputs]).cpu().numpy()
        phiinv_disp_list_4 = torch.cat([output["phiinv_disp_list"][4] for output in outputs]).cpu().numpy()

        if TSteps != 5:
            phiinv_disp_list_5 = torch.cat([output["phiinv_disp_list"][5] for output in outputs]).cpu().numpy()
            phiinv_disp_list_6 = torch.cat([output["phiinv_disp_list"][6] for output in outputs]).cpu().numpy()
            phiinv_disp_list_7 = torch.cat([output["phiinv_disp_list"][7] for output in outputs]).cpu().numpy()
            phiinv_disp_list_8 = torch.cat([output["phiinv_disp_list"][8] for output in outputs]).cpu().numpy()
            phiinv_disp_list_9 = torch.cat([output["phiinv_disp_list"][9] for output in outputs]).cpu().numpy()

        #dfm list
        dfmSrc_list_0 = torch.cat([output["dfmSrc_list"][0] for output in outputs]).cpu().numpy()
        dfmSrc_list_1 = torch.cat([output["dfmSrc_list"][1] for output in outputs]).cpu().numpy()
        dfmSrc_list_2 = torch.cat([output["dfmSrc_list"][2] for output in outputs]).cpu().numpy()
        dfmSrc_list_3 = torch.cat([output["dfmSrc_list"][3] for output in outputs]).cpu().numpy()
        dfmSrc_list_4 = torch.cat([output["dfmSrc_list"][4] for output in outputs]).cpu().numpy()
        if TSteps != 5:
            dfmSrc_list_5 = torch.cat([output["dfmSrc_list"][5] for output in outputs]).cpu().numpy()
            dfmSrc_list_6 = torch.cat([output["dfmSrc_list"][6] for output in outputs]).cpu().numpy()
            dfmSrc_list_7 = torch.cat([output["dfmSrc_list"][7] for output in outputs]).cpu().numpy()
            dfmSrc_list_8 = torch.cat([output["dfmSrc_list"][8] for output in outputs]).cpu().numpy()
            dfmSrc_list_9 = torch.cat([output["dfmSrc_list"][9] for output in outputs]).cpu().numpy()




    """ # for visualization
    #print(srcs.shape, tars.shape, preds.shape, VecF_0.shape, phiinv_disp_list_0.shape, JabDet.shape, dfmSrclabels.shape) 
    #print(dir)
    sreimg = sitk.GetImageFromArray(srcs[0, 0])
    tarimg = sitk.GetImageFromArray(tars[0, 0])
    predimg = sitk.GetImageFromArray(preds[0, 0])

    dfmSrclabels = clear_labels(dfmSrclabels)
    srclabels = clear_labels(srclabels)
    tarlabels = clear_labels(tarlabels)

    dfmSrclabelimg = sitk.GetImageFromArray(dfmSrclabels[0, 0])
    srclabelimg = sitk.GetImageFromArray(srclabels[0, 0])
    tarlabelimg = sitk.GetImageFromArray(tarlabels[0, 0])

    sitk.WriteImage(sreimg, f"{dir}/src.nii.gz")
    sitk.WriteImage(tarimg, f"{dir}/tar.nii.gz")
    sitk.WriteImage(predimg, f"{dir}/pred.nii.gz")
    sitk.WriteImage(dfmSrclabelimg, f"{dir}/dfmSrclabel.nii.gz")
    sitk.WriteImage(srclabelimg, f"{dir}/srclabel.nii.gz")
    sitk.WriteImage(tarlabelimg, f"{dir}/tarlabel.nii.gz")
    # (1, 1, 128, 128, 128) (1, 1, 128, 128, 128) (1, 1, 128, 128, 128) (1, 128, 128, 128, 3) (1, 3, 128, 128, 128) (1, 1, 128, 128, 128)
    assert 3>333 """




    #(100, 1, 32, 32)         (100, 1, 32, 32)        (100, 1, 32, 32)        (100, 32, 32, 3)        (100, 2, 32, 32)
    #(100, 1, 128, 128, 128)  (100, 1, 128, 128, 128) (100, 1, 128, 128, 128) (100, 128, 128, 128, 3) (100, 3, 128, 128, 128)
    # (10, 1, 128, 128, 128)  (10, 1, 128, 128, 128)  (10, 1, 128, 128, 128)  (10, 128, 128, 128, 3)  (10, 3, 128, 128, 128)


    # save np.save(f"{dir}/JabDetSlice_{vie}.npy", JabDetSlice)
    # save srcs tars preds


    # Show different views of the same data  
    # !!!!!!!! Select the first view
    """ if len(srcs.shape) == 5: #3D
        vie = 0
        select_idx = 59
        srcs = srcs[:, :, select_idx]
        tars = tars[:, :, select_idx]
        preds = preds[:, :, select_idx]
        JabDet = JabDet[:, :, select_idx]
        
        VecF_0 = VecF_0[:, select_idx] #or VecF_0 = VecF_0[:, :, select_idx] or VecF_0 = VecF_0[:, :, :, select_idx]
        VecF_1 = VecF_1[:, select_idx]
        VecF_2 = VecF_2[:, select_idx]
        VecF_3 = VecF_3[:, select_idx]
        VecF_4 = VecF_4[:, select_idx]
        if TSteps != 5:
            VecF_5 = VecF_5[:, select_idx]
            VecF_6 = VecF_6[:, select_idx]
            VecF_7 = VecF_7[:, select_idx]
            VecF_8 = VecF_8[:, select_idx]
            VecF_9 = VecF_9[:, select_idx]

        #or phiinv_disp_list_0 = phiinv_disp_list_0[:, ::2, : , select_idx]  or phiinv_disp_list_0 = phiinv_disp_list_0[:, 0:2, :, :, select_idx]
        phiinv_disp_list_0 = phiinv_disp_list_0[:, 1:, select_idx] 
        phiinv_disp_list_1 = phiinv_disp_list_1[:, 1:, select_idx]
        phiinv_disp_list_2 = phiinv_disp_list_2[:, 1:, select_idx]
        phiinv_disp_list_3 = phiinv_disp_list_3[:, 1:, select_idx]
        phiinv_disp_list_4 = phiinv_disp_list_4[:, 1:, select_idx]
        if TSteps != 5:
            phiinv_disp_list_5 = phiinv_disp_list_5[:, 1:, select_idx]
            phiinv_disp_list_6 = phiinv_disp_list_6[:, 1:, select_idx]
            phiinv_disp_list_7 = phiinv_disp_list_7[:, 1:, select_idx]
            phiinv_disp_list_8 = phiinv_disp_list_8[:, 1:, select_idx]
            phiinv_disp_list_9 = phiinv_disp_list_9[:, 1:, select_idx]

        dfmSrc_list_0 = dfmSrc_list_0[:, :, select_idx]
        dfmSrc_list_1 = dfmSrc_list_1[:, :, select_idx]
        dfmSrc_list_2 = dfmSrc_list_2[:, :, select_idx]
        dfmSrc_list_3 = dfmSrc_list_3[:, :, select_idx]
        dfmSrc_list_4 = dfmSrc_list_4[:, :, select_idx]
        if TSteps != 5:
            dfmSrc_list_5 = dfmSrc_list_5[:, :, select_idx]
            dfmSrc_list_6 = dfmSrc_list_6[:, :, select_idx]
            dfmSrc_list_7 = dfmSrc_list_7[:, :, select_idx]
            dfmSrc_list_8 = dfmSrc_list_8[:, :, select_idx]
            dfmSrc_list_9 = dfmSrc_list_9[:, :, select_idx] """




    # Show different views of the same data  
    # !!!!!!!! Select the second view
    if len(srcs.shape) == 5: #3D
        select_idx = 64
        vie = 1
        imagesize = srcs.shape[-1] #128
        select_idx=int(imagesize/2-3*(imagesize/64))
        select_idx = 59
        srcs = srcs[:, :, :, select_idx]
        tars = tars[:, :, :, select_idx]
        preds = preds[:, :, :, select_idx]
        JabDet = JabDet[:, :, :, select_idx]
    
        VecF_0 = VecF_0[:, :, select_idx] #or VecF_0 = VecF_0[:, :, select_idx] or VecF_0 = VecF_0[:, :, :, select_idx]
        VecF_1 = VecF_1[:, :, select_idx]
        VecF_2 = VecF_2[:, :, select_idx]
        VecF_3 = VecF_3[:, :, select_idx]
        VecF_4 = VecF_4[:, :, select_idx]
        if TSteps != 5:
            VecF_5 = VecF_5[:, :, select_idx]
            VecF_6 = VecF_6[:, :, select_idx]
            VecF_7 = VecF_7[:, :, select_idx]
            VecF_8 = VecF_8[:, :, select_idx]
            VecF_9 = VecF_9[:, :, select_idx]
        
        #or phiinv_disp_list_0 = phiinv_disp_list_0[:, ::2, :, select_idx]  or phiinv_disp_list_0 = phiinv_disp_list_0[:, 0:2, :, :, select_idx]
        phiinv_disp_list_0 = phiinv_disp_list_0[:, ::2, :, select_idx] 
        phiinv_disp_list_1 = phiinv_disp_list_1[:, ::2, :, select_idx]
        phiinv_disp_list_2 = phiinv_disp_list_2[:, ::2, :, select_idx]
        phiinv_disp_list_3 = phiinv_disp_list_3[:, ::2, :, select_idx]
        phiinv_disp_list_4 = phiinv_disp_list_4[:, ::2, :, select_idx]
        if TSteps != 5:
            phiinv_disp_list_5 = phiinv_disp_list_5[:, ::2, :, select_idx]
            phiinv_disp_list_6 = phiinv_disp_list_6[:, ::2, :, select_idx]
            phiinv_disp_list_7 = phiinv_disp_list_7[:, ::2, :, select_idx]
            phiinv_disp_list_8 = phiinv_disp_list_8[:, ::2, :, select_idx]
            phiinv_disp_list_9 = phiinv_disp_list_9[:, ::2, :, select_idx]

        dfmSrc_list_0 = dfmSrc_list_0[:, :, :, select_idx]
        dfmSrc_list_1 = dfmSrc_list_1[:, :, :, select_idx]
        dfmSrc_list_2 = dfmSrc_list_2[:, :, :, select_idx]
        dfmSrc_list_3 = dfmSrc_list_3[:, :, :, select_idx]
        dfmSrc_list_4 = dfmSrc_list_4[:, :, :, select_idx]
        if TSteps != 5:
            dfmSrc_list_5 = dfmSrc_list_5[:, :, :, select_idx]
            dfmSrc_list_6 = dfmSrc_list_6[:, :, :, select_idx]
            dfmSrc_list_7 = dfmSrc_list_7[:, :, :, select_idx]
            dfmSrc_list_8 = dfmSrc_list_8[:, :, :, select_idx]
            dfmSrc_list_9 = dfmSrc_list_9[:, :, :, select_idx]




    

    # Show different views of the same data  
    # !!!!!!!! Select the third view
    # if len(srcs.shape) == 5: #3D
    #     vie = 2
    #     select_idx = 59
    #     srcs = srcs[:, :, :, :, select_idx]
    #     tars = tars[:, :, :, :, select_idx]
    #     preds = preds[:, :, :, :, select_idx]
    #     JabDet = JabDet[:, :, :, :, select_idx]
        
    #     VecF_0 = VecF_0[:, :, :, select_idx] #or VecF_0 = VecF_0[:, :, select_idx] or VecF_0 = VecF_0[:, :, :, select_idx]
    #     VecF_1 = VecF_1[:, :, :, select_idx]
    #     VecF_2 = VecF_2[:, :, :, select_idx]
    #     VecF_3 = VecF_3[:, :, :, select_idx]
    #     VecF_4 = VecF_4[:, :, :, select_idx]
    #     if TSteps != 5:
    #         VecF_5 = VecF_5[:, :, :, select_idx]
    #         VecF_6 = VecF_6[:, :, :, select_idx]
    #         VecF_7 = VecF_7[:, :, :, select_idx]
    #         # VecF_8 = VecF_8[:, :, :, select_idx]
    #         # VecF_9 = VecF_9[:, :, :, select_idx]

    #     #or phiinv_disp_list_0 = phiinv_disp_list_0[:, ::2, : , select_idx]  or phiinv_disp_list_0 = phiinv_disp_list_0[:, 0:2, :, :, select_idx]
    #     phiinv_disp_list_0 = phiinv_disp_list_0[:, 0:2, :, :, select_idx] 
    #     phiinv_disp_list_1 = phiinv_disp_list_1[:, 0:2, :, :, select_idx]
    #     phiinv_disp_list_2 = phiinv_disp_list_2[:, 0:2, :, :, select_idx]
    #     phiinv_disp_list_3 = phiinv_disp_list_3[:, 0:2, :, :, select_idx]
    #     phiinv_disp_list_4 = phiinv_disp_list_4[:, 0:2, :, :, select_idx]
    #     if TSteps != 5:
    #         phiinv_disp_list_5 = phiinv_disp_list_5[:, 0:2, :, :, select_idx]
    #         phiinv_disp_list_6 = phiinv_disp_list_6[:, 0:2, :, :, select_idx]
    #         phiinv_disp_list_7 = phiinv_disp_list_7[:, 0:2, :, :, select_idx]
    #         # phiinv_disp_list_8 = phiinv_disp_list_8[:, 0:2, :, :, select_idx]
    #         # phiinv_disp_list_9 = phiinv_disp_list_9[:, 0:2, :, :, select_idx]

    #     dfmSrc_list_0 = dfmSrc_list_0[:, :, :, :, select_idx]
    #     dfmSrc_list_1 = dfmSrc_list_1[:, :, :, :, select_idx]
    #     dfmSrc_list_2 = dfmSrc_list_2[:, :, :, :, select_idx]
    #     dfmSrc_list_3 = dfmSrc_list_3[:, :, :, :, select_idx]
    #     dfmSrc_list_4 = dfmSrc_list_4[:, :, :, :, select_idx]
    #     if TSteps != 5:
    #         dfmSrc_list_5 = dfmSrc_list_5[:, :, :, :, select_idx]
    #         dfmSrc_list_6 = dfmSrc_list_6[:, :, :, :, select_idx]
    #         dfmSrc_list_7 = dfmSrc_list_7[:, :, :, :, select_idx]
    #         dfmSrc_list_8 = dfmSrc_list_8[:, :, :, :, select_idx]
    #         dfmSrc_list_9 = dfmSrc_list_9[:, :, :, :, select_idx]




    

    if TSteps != 5:
        VecF_list = [VecF_0, VecF_1, VecF_2, VecF_3, VecF_4, VecF_5, VecF_6, VecF_7, VecF_8, VecF_9]
        phiinv_disp_list_list = [phiinv_disp_list_0, phiinv_disp_list_1, phiinv_disp_list_2, phiinv_disp_list_3, phiinv_disp_list_4, phiinv_disp_list_5, phiinv_disp_list_6, phiinv_disp_list_7, phiinv_disp_list_8, phiinv_disp_list_9]
        dfmSrc_list_list = [dfmSrc_list_0, dfmSrc_list_1, dfmSrc_list_2, dfmSrc_list_3, dfmSrc_list_4, dfmSrc_list_5, dfmSrc_list_6, dfmSrc_list_7, dfmSrc_list_8, dfmSrc_list_9]

        # VecF_list = [VecF_0, VecF_1, VecF_2, VecF_3, VecF_4, VecF_5, VecF_6, VecF_7]
        # phiinv_disp_list_list = [phiinv_disp_list_0, phiinv_disp_list_1, phiinv_disp_list_2, phiinv_disp_list_3, phiinv_disp_list_4, phiinv_disp_list_5, phiinv_disp_list_6, phiinv_disp_list_7]
        # dfmSrc_list_list = [dfmSrc_list_0, dfmSrc_list_1, dfmSrc_list_2, dfmSrc_list_3, dfmSrc_list_4, dfmSrc_list_5, dfmSrc_list_6, dfmSrc_list_7]

    else:
        VecF_list = [VecF_0, VecF_1, VecF_2, VecF_3, VecF_4]
        phiinv_disp_list_list = [phiinv_disp_list_0, phiinv_disp_list_1, phiinv_disp_list_2, phiinv_disp_list_3, phiinv_disp_list_4]
        dfmSrc_list_list = [dfmSrc_list_0, dfmSrc_list_1, dfmSrc_list_2, dfmSrc_list_3, dfmSrc_list_4]
    
    
    mse_list_list = []
    # #print(tars.shape, dfmSrc_list_0.shape, preds.shape)  #(1, 1, 32, 32) (1, 1, 32, 32)  (1, 1, 32, 32)
    # for i in range(tars.shape[0]): #循环100次
    #     mse_list = []
    #     for t in range(TSteps):
    #         dfm =  dfmSrc_list_list[t]
    #         mse_list.append(mse_metric(torch.from_numpy(dfm[i:i+1]), torch.from_numpy(preds[i:i+1])))

    # for i in range(tars.shape[0]): #循环100次
    #     mse_list = []
    #     mse = mse_metric(torch.from_numpy(tars), torch.from_numpy(srcs))
    #     mse_list.append(mse.item())
    #     for t in range(TSteps):
    #         dfm =  dfmSrc_list_list[t]
    #         mse = mse_metric(torch.from_numpy(dfm[i:i+1]), torch.from_numpy(tars[i:i+1]))
    #         mse_list.append(mse.item())
    
    # #print("mse_list\n", mse_list)   
    # mse_list_string = ' '.join(map(str, mse_list))+"\n"
    # file = open(f"{dir}/lddmm_plot_geodesic_path_mse.txt", "a+")
    # file.write(f"{mse_list_string}\n")


    # #save phiinv_disp_list_list as numpy
    # np.save("/home/nellie/code/cvpr/Project/DynamiCrafter/2025_SAVE_ROOT_DIR/2025training_mnist_v2.1/phiinv_disp_list_list.npy", np.stack(phiinv_disp_list_list, axis=0))
    # assert 3>123
    if saveRes:
        nums = srcs.shape[0]
        # save srcss and tars as npy  #delete
        np.save(f"{savedir}/10_Steps{TSteps}_{nums}_LDDMM_dfmSrclabels.npy", dfmSrclabels)
        # np.save(f"{savedir}/10_Steps{TSteps}_{nums}_LDDMM_tarlabels.npy", tarlabels)
        
        for i in range(TSteps):  
            np.save(f"{savedir}/10_Steps{TSteps}_{nums}_LDDMM_dfmSrc_list_{i}.npy", dfmSrc_list_list[i])
            np.save(f"{savedir}/10_Steps{TSteps}_{nums}_LDDMM_phiinv_disp_list_{i}.npy", phiinv_disp_list_list[i])
    
    

    return srcs, tars, preds, VecF_list, phiinv_disp_list_list, dfmSrc_list_list, dfmSrclabels, vie, JabDet


def plot_dfm_process(outputs, nrows_list=['dfmSrc','phiinv',"v"], ongpuGrad=False, inshape=None, mse_metric=None, TSteps=10, showTSteps=[0, 1, 3, 5, 7, 9], dir="", saveRes=True):
    if not os.path.exists(dir) and dir != "":
        os.makedirs(dir)

    JabDet = get_jacobian_det_for_all(outputs, ongpuGrad, inshape)
    srcs, tars, preds, VecF_list, phiinv_disp_list_list, dfmSrc_list_list, dfmSrclabels, vie, JabDetSlice = collect_outputs(outputs,ongpuGrad,TSteps,dir=dir, mse_metric=mse_metric,saveRes=saveRes, JabDet=JabDet)
    
    # print(srcs.shape, tars.shape,       preds.shape,     VecF_list[0].shape, phiinv_disp_list_list[0].shape, dfmSrc_list_list[0].shape)
    # (107, 1, 64, 64)  (107, 1, 64, 64) (107, 1, 64, 64)  (107, 64, 64, 3)    (107, 2, 64, 64)                (107, 1, 64, 64)
    # assert 1>222
    imgsize = int(srcs.shape[-1]/3); imgsize = max(32, imgsize)
    # 
    timesteps = showTSteps
    batch_size = min(25, srcs.shape[0])


    nrows_per_exmp = len(nrows_list)
    nrows = batch_size*nrows_per_exmp
    ncols = len(timesteps)+3

    #plt.subplots(nrows, ncols)
    #axes:(nrows, ncols)
    #figsize=(width, height)

    # 创建图形
    basesize = 2.5
    basesize = 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*basesize, nrows*basesize))  # 每行 3  batch_size 行
    plt.subplots_adjust(hspace=0.05)  # 这里的 0.05 代表行间距，值越小间距越小
    plt.subplots_adjust(wspace=0.05)  # 这里的 0.05 代表行间距，值越小间距越小

    for iix in range(batch_size):
        # axes_y = {
        #     "0": nrows_per_exmp*i,
        #     "1": nrows_per_exmp*i+1,
        #     "2": nrows_per_exmp*i+2
        # }
        i = iix
        axes_y = [axes[nrows_per_exmp*i+nrow] for nrow in range(len(nrows_list))]
        # i = iix+60

        # source and target
        ax_src = axes_y[0][0]
        ax_src.imshow(srcs[i, 0], cmap="gray", vmin=0, vmax=1)  # 选择第 0 通道
        ax_src.axis("off")
        ax_src.set_title(f"{i+1} - Source", fontsize=16)
        
        ax_tar = axes_y[0][1]
        ax_tar.imshow(tars[i, 0], cmap="gray", vmin=0, vmax=1)  # 选择第 0 通道
        ax_tar.axis("off")
        ax_tar.set_title("Target", fontsize=16)

        # dfmSrc
        ax_pred = axes_y[0][2]
        ax_pred.imshow(preds[i, 0], cmap="gray", vmin=0, vmax=1)  # 选择第 0 通道
        ax_pred.axis("off")
        ax_pred.set_title("DfmSrc", fontsize=16)

        #clear empty space
        for nrow in range(1, len(nrows_list)):
            axes_y[nrow][0].cla(); axes_y[nrow][0].axis("off")
            axes_y[nrow][1].cla(); axes_y[nrow][1].axis("off")
            axes_y[nrow][2].cla(); axes_y[nrow][2].axis("off")

        #plot velicities/phiinvs/dfmsrcs
        # nrows_list = ["v", "phiinv", "dfmSrc"]
        

        idx_v = nrows_list.index("v")
        idx_phiinv = nrows_list.index("phiinv")
        idx_dfmSrc = nrows_list.index("dfmSrc")

        #1.plot velocities
        axes_y_v = axes_y[idx_v]  
        for idx, timestep in enumerate(timesteps):#timesteps = [0, 1, 3, 5, 7, 9]
            # VecF_list
            v = VecF_list[timestep][i]  #(32, 32, 3)
            # #print(v.shape, v.min(), v.max(), v.mean(), v.dtype)
            # assert 1>222
            # v = VecF_list[timestep][i]
            # v = (v - v.min()) / (v.max() - v.min())
            axes_y_v[idx+3].imshow(v)

            # axes_y_v[idx+3].imshow(VecF_list[timestep][i], vmin=0, vmax=1)
            axes_y_v[idx+3].axis("off")
            # axes_y_v[idx+3].set_title("V_{}".format(timestep), fontsize=16)

        #2.plot phiinvs
        axes_y_phiinv = axes_y[idx_phiinv]
        for idx, timestep in enumerate(timesteps):#timesteps = [0, 1, 3, 5, 7, 9]
            # Mgridplot(phiinv_disp_list_list[timestep][i:i+1], axes_y_phiinv[idx+3],  imgsize, imgsize, displacement = True, IMG=tars[i, 0])
            Mgridplot(phiinv_disp_list_list[timestep][i:i+1], axes_y_phiinv[idx+3],  imgsize, imgsize, displacement = True)

        #3.plot dfmSrcs
        axes_y_dfmSrc = axes_y[idx_dfmSrc]
        for idx, timestep in enumerate(timesteps):#timesteps = [0, 1, 3, 5, 7, 9]
            axes_y_dfmSrc[idx+3].imshow(dfmSrc_list_list[timestep][i,0], cmap="gray", vmin=0, vmax=1)
            axes_y_dfmSrc[idx+3].set_title("T_{}".format(timestep), fontsize=16)
            axes_y_dfmSrc[idx+3].axis("off")


    # 调整子图布局
    plt.tight_layout()
    

    # #save JabDetSlice
    # np.save(f"{dir}/JabDetSlice_{vie}.npy", JabDetSlice)


    # savep = f"{dir}/visualization_{vie}.png"
    # #print(f"save the plot in {savep}")
    # plt.savefig(savep,bbox_inches='tight',dpi=200)

    plt.show()
    plt.close()




def _collect_one_step(
    outputs: List[Dict],
    key: str,
    t: int,
    *,
    cat_dim: int = 0
) -> np.ndarray:
    """从多份 outputs 中取同一时间步的张量并 cat，再转 numpy。"""
    print(type(outputs[0][key]), len(outputs[0][key]), t)
    # assert 4>987
    tensors = [out[key][t] for out in outputs]
    return torch.cat(tensors, dim=cat_dim).cpu().numpy()

def collect_series(
    outputs: List[Dict],
    key: str,
    *,
    steps: Optional[int] = None,
    cat_dim: int = 0,
    post: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> List[np.ndarray]:
    """
    收集一个 key 的时间序列：
    - outputs: DataLoader 各 batch 的输出列表（每个元素是 dict）
    - key: 比如 "VecF_List" / "phiinv_disp_list" / "dfmSrc_list"
    - steps: 时间步数；不传则从 outputs[0][key] 长度推断
    - cat_dim: torch.cat 的维度（默认 0）
    - post: 可选的后处理函数，接受 np.ndarray，返回 np.ndarray
    返回：长度为 steps 的列表，每个元素是该时间步 cat 后的 numpy 数组
    """
    if steps is None:
        steps = len(outputs[0][key])
    series = []
    for t in steps:
        arr = _collect_one_step(outputs, key, t, cat_dim=cat_dim)
        if post is not None:
            arr = post(arr)
        series.append(arr)
    return series

# ===== 使用示例 =====
# 假设你已有：outputs, TSteps, add_zero_channel_and_norm






def collect_outputs_series(outputs, showTSteps=[1,3,5,7,9,11,13]):
    
    srcs = torch.cat([output["src"] for output in outputs]).cpu().numpy()
    srclabels = torch.cat([output["srclabel"] for output in outputs]).cpu().numpy()
    # print(srcs.shape)  #(12, 1, 64, 64)
    # if len(srcs.shape) == 5: #3D
    #     srclabels = torch.cat([output["srclabel"] for output in outputs]).cpu().numpy()
    #     dfmSrclabels = torch.cat([output["dfmSrclabel"] for output in outputs]).cpu().numpy()  #(100, 1, 128, 128, 128)
    #     tarlabels = torch.cat([output["tarlabel"] for output in outputs]).cpu().numpy()
    # elif len(srcs.shape) == 4: #2D
    #     dfmSrclabels = None


    # VecF（带后处理）
    VecF_list = collect_series(
        outputs,
        "VecF_List",
        steps=showTSteps,                             # 可传可不传
        post=lambda x: add_zero_channel_and_norm(x)
    )
    # phiinv_disp_list
    phiinv_disp_list_list = collect_series(
        outputs,
        "phiinv_disp_list",
        steps=showTSteps
    )
    # dfmSrc_list
    dfmSrc_list_list = collect_series(
        outputs,
        "dfmSrc_list",
        steps=showTSteps
    )
    #tar_list
    tar_list_list = collect_series(
        outputs,
        "tar_list",
        steps=showTSteps
    )
    #dfmSrclabel_list
    dfmSrclabel_list_list = collect_series(
        outputs,
        "dfmSrclabel_list",     
        steps=showTSteps
    )
    #tarlabel_list
    tarlabel_list_list = collect_series(
        outputs,
        "tarlabel_list",     
        steps=showTSteps
    )

    return srcs, srclabels, tar_list_list, dfmSrc_list_list, VecF_list, phiinv_disp_list_list, dfmSrc_list_list, dfmSrclabel_list_list, tarlabel_list_list

def plot_dfm_process_for_series(outputs, nrows_list=['tar','dfmSrc',"disp"], showTSteps=[1,3,5,7,9,11,13], dir="", savePath=None):
    nrows_list=['tar', "tarlabel", 'dfmSrc', "dfmSrclabel"]
    nrows_list=['tar', "tarlabel", 'dfmSrc', "dfmSrclabel", "disp", "diff"]

    idx_tar = nrows_list.index("tar") if "tar" in nrows_list else None
    idx_dfmSrc = nrows_list.index("dfmSrc") if "dfmSrc" in nrows_list else None
    idx_disp = nrows_list.index("disp") if "disp" in nrows_list else None
    idx_tarlabel = nrows_list.index("tarlabel") if "tarlabel" in nrows_list else None
    idx_dfmSrclabel = nrows_list.index("dfmSrclabel") if "dfmSrclabel" in nrows_list else None
    idx_diff = nrows_list.index("diff") if "diff" in nrows_list else None


    if not os.path.exists(dir) and dir != "":
        os.makedirs(dir)
    srcs, srclabels, tar_list_list, dfmSrc_list_list, VecF_list, phiinv_disp_list_list, dfmSrc_list_list, dfmSrclabel_list_list, tarlabel_list_list = collect_outputs_series(outputs,showTSteps=showTSteps)
    # print(srcs.shape, tars.shape,       preds.shape,     VecF_list[0].shape, phiinv_disp_list_list[0].shape, dfmSrc_list_list[0].shape)
    # (107, 1, 64, 64)  (107, 1, 64, 64) (107, 1, 64, 64)  (107, 64, 64, 3)    (107, 2, 64, 64)                (107, 1, 64, 64)
    # assert 1>222

    diff_dfm_tar_list = [np.abs(dfmSrclabel_list_list[idx]-tarlabel_list_list[idx]) for idx in range(len(dfmSrclabel_list_list)) ]


    imgsize = int(srcs.shape[-1]/3); imgsize = max(32, imgsize)
    timesteps = showTSteps
    batch_size = min(20, srcs.shape[0])

    left_col_nums = 1
    nrows_per_exmp = len(nrows_list)
    nrows = batch_size*nrows_per_exmp
    ncols = len(timesteps)+left_col_nums
    #plt.subplots(nrows, ncols)
    #axes:(nrows, ncols)
    #figsize=(width, height)

    # 创建图形
    basesize = 2.5
    basesize = 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*basesize, nrows*basesize))  # 每行 3  batch_size 行
    plt.subplots_adjust(hspace=0.05)  # 这里的 0.05 代表行间距，值越小间距越小
    plt.subplots_adjust(wspace=0.05)  # 这里的 0.05 代表行间距，值越小间距越小

    for iix in range(batch_size):
        # axes_y = {
        #     "0": nrows_per_exmp*i,
        #     "1": nrows_per_exmp*i+1,
        #     "2": nrows_per_exmp*i+2
        # }
        i = iix
        axes_y = [axes[nrows_per_exmp*i+nrow] for nrow in range(len(nrows_list))]
        # i = iix+60

        # source and target
        ax_src = axes_y[0][0]
        ax_src.imshow(srcs[i, 0], cmap="gray", vmin=0, vmax=1)  # 选择第 0 通道
        ax_src.axis("off")
        ax_src.set_title(f"{i+1} - Source", fontsize=20)


        
        ax_srclabel = axes_y[1][0]
        ax_srclabel.imshow(srcs[i, 0], cmap="gray", vmin=0, vmax=1)  # 选择第 0 通道
        ax_srclabel.imshow(srclabels[i, 0], cmap="gray", alpha=0.99)  # 选择第 0 通道
        ax_srclabel.axis("off")
        ax_srclabel.set_title("Source_Label", fontsize=20)


        
        #clear empty space
        for nrow in range(2, len(nrows_list)):
            axes_y[nrow][0].cla(); axes_y[nrow][0].axis("off")
        # idx_tar = nrows_list.index("tar")
        # idx_dfmSrc = nrows_list.index("dfmSrc")
        # idx_disp = nrows_list.index("disp")

        # #1.plot velocity
        # axes_y_v = axes_y[idx_tar]  
        # for idx, timestep in enumerate(timesteps):#timesteps = [0, 1, 3, 5, 7, 9]
        #     # VecF_list
        #     v = VecF_list[idx][i]  #(32, 32, 3)
        #     # print(v.shape, v.min(), v.max(), v.mean(), v.dtype)
        #     # assert 1>222
        #     # v = VecF_list[timestep][i]
        #     # v = (v - v.min()) / (v.max() - v.min())
        #     axes_y_v[idx+left_col_nums].imshow(v)

        #     # axes_y_v[idx+3].imshow(VecF_list[timestep][i], vmin=0, vmax=1)
        #     axes_y_v[idx+left_col_nums].axis("off")
        #     # axes_y_v[idx+3].set_title("V_{}".format(timestep), fontsize=16)


        #1.plot idx_tar
        if idx_tar is not None:
            axes_idx_tar = axes_y[idx_tar]
            for idx, timestep in enumerate(timesteps):#timesteps = [0, 1, 3, 5, 7, 9]
                axes_idx_tar[idx+left_col_nums].imshow(tar_list_list[idx][i,0], cmap="gray", vmin=0, vmax=1)
                axes_idx_tar[idx+left_col_nums].set_title("Tar_{}".format(timestep), fontsize=20)
                axes_idx_tar[idx+left_col_nums].axis("off")

        #2.plot idx_dfmSrc
        if idx_dfmSrc is not None:
            axes_y_dfmSrc = axes_y[idx_dfmSrc]
            for idx, timestep in enumerate(timesteps):#timesteps = [0, 1, 3, 5, 7, 9]
                axes_y_dfmSrc[idx+left_col_nums].imshow(dfmSrc_list_list[idx][i,0], cmap="gray", vmin=0, vmax=1)
                axes_y_dfmSrc[idx+left_col_nums].set_title("Dfm_{}".format(timestep), fontsize=20)
                axes_y_dfmSrc[idx+left_col_nums].axis("off")
        
        #3.plot phiinv
        if idx_disp is not None:
            axes_y_phiinv = axes_y[idx_disp]
            for idx, timestep in enumerate(timesteps):#timesteps = [0, 1, 3, 5, 7, 9]
                # Mgridplot(phiinv_disp_list_list[timestep][i:i+1], axes_y_phiinv[idx+3],  imgsize, imgsize, displacement = True, IMG=tars[i, 0])
                Mgridplot(phiinv_disp_list_list[idx][i:i+1], axes_y_phiinv[idx+left_col_nums],  imgsize, imgsize, displacement = True)


        #4.plot idx_tarlabel
        if idx_tarlabel is not None:
            axes_y_tarlabel = axes_y[idx_tarlabel]
            for idx, timestep in enumerate(timesteps):#timesteps = [0, 1, 3, 5, 7, 9]
                # overlay tarlabel on tar
                axes_y_tarlabel[idx+left_col_nums].imshow(tar_list_list[idx][i,0], cmap="gray", vmin=0, vmax=1)
                axes_y_tarlabel[idx+left_col_nums].imshow(tarlabel_list_list[idx][i,0], cmap="gray", alpha=0.99)

                axes_y_tarlabel[idx+left_col_nums].set_title("TarLabel_{}".format(timestep), fontsize=20)
                axes_y_tarlabel[idx+left_col_nums].axis("off")
        #5.plot idx_dfmSrclabel
        if idx_dfmSrclabel is not None:
            axes_y_dfmSrclabel = axes_y[idx_dfmSrclabel]
            for idx, timestep in enumerate(timesteps):#timesteps = [0,
                axes_y_dfmSrclabel[idx+left_col_nums].imshow(dfmSrc_list_list[idx][i,0], cmap="gray", vmin=0, vmax=1)
                axes_y_dfmSrclabel[idx+left_col_nums].imshow(dfmSrclabel_list_list[idx][i,0], cmap="gray", alpha=0.99)

                axes_y_dfmSrclabel[idx+left_col_nums].set_title("DfmSrcLabel_{}".format(timestep), fontsize=20)
                axes_y_dfmSrclabel[idx+left_col_nums].axis("off") 

        #6.plot difference
        if idx_diff is not None:
            axes_y_diff = axes_y[idx_diff]
            for idx, timestep in enumerate(timesteps):#timesteps = [0,
                axes_y_diff[idx+left_col_nums].imshow(diff_dfm_tar_list[idx][i,0], cmap="gray", vmin=0, vmax=1)
                axes_y_diff[idx+left_col_nums].imshow(diff_dfm_tar_list[idx][i,0], cmap="gray", alpha=0.99)

                axes_y_diff[idx+left_col_nums].set_title("DfmSrcLabel_{}".format(timestep), fontsize=20)
                axes_y_diff[idx+left_col_nums].axis("off") 


    
    # 调整子图布局
    plt.tight_layout()
    print(savePath)
    # #save JabDetSlice
    # np.save(f"{dir}/JabDetSlice_{vie}.npy", JabDetSlice)
    # savep = f"{dir}/visualization_{vie}.png"
    # print(f"save the plot in {savep}")
    plt.savefig(savePath,bbox_inches='tight',dpi=200)
    # plt.show()
    plt.close()





















def plot_boxplot_by_textFiles(textFiles_List, methods_List, saveP = "", bioname=['WM', 'CerebralC', 'Ven', 'None', 'CerebellumC', 'Tha', 'Caud', 'Puta', 'Hipp', 'Stem', 'None', 'NoNe']):
    #  [1, 2, 3, 5, 6, 7, 8, 9, 14, 13]
    # bioname = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    # bioname = ['WM', 'CerebralC', 'Ven', 'None', 'CerebellumC', 'Tha', 'Caud', 'Puta', 'Hipp', 'Stem', 'None', 'NoNe']
    # bioname = ['Step5', 'Step6', 'Step7', 'Step8', 'Step9', 'Step10', 'Caud', 'Puta', 'Hipp', 'Stem', 'None', 'NoNe']
    # #print("bioname: ", bioname)
    # #print("bioname: ", bioname)
    # #print("bioname: ", bioname)




    module_color_list = ['r', 'g', '#000000', 'b', 'c', 'm', 'y', 'k', '#FF5733', '#33FF57', '#5733FF', '#FF33F7', '#00FFCC', '#9900FF']
    module_color_list2 = [color for color in module_color_list for _ in range(2)]

    gap = 0.7
    widths = 0.5
    plt.figure(figsize=(12,4))
    
    dice_file_list = []
    positions_list = []

    for idx, path in enumerate(textFiles_List):
        # #print("\n\n",idx)
        file = open(path)
        lines = file.readlines()   ##length=50
        # lines = lines[:100]
        dice_file_float_arr_cur = [list(map(float, dice_cur_str.strip('\n').split(' '))) for dice_cur_str in lines]

        dataCur = np.array(dice_file_float_arr_cur)
        #print(dataCur.shape, "Avg-", np.mean(dataCur, axis=0) ,path, "\n")

        dice_file_list.append(np.array(dice_file_float_arr_cur))
        # #print(idx,"-file:  ",dice_file_list[-1].shape)
        positions_list.append(idx*gap)

    dice_by_biology = np.stack(dice_file_list,axis=0)   #4,50,11
    # #print("All-file:  ",dice_by_biology.shape)
    x_name_by_biology = np.arange(dice_by_biology.shape[-1])


    x_name_by_biology = [bioname[i] for i in x_name_by_biology]
    x_name_by_biology_new = []
    x_name_by_biology_position = []




    ### start plot the box
    passNum = 0
    for bio_idx in range(dice_by_biology.shape[-1]):
        if (bioname[bio_idx] == 'None'):
            passNum += 1
            continue    
        x_name_by_biology_new.append(bioname[bio_idx])
        base = (bio_idx-passNum)*len(textFiles_List)+1
        x_name_by_biology_position.append(base)

        data = dice_by_biology[...,bio_idx]
        data = np.transpose(data, (1,0))
        # #print(data.shape)
        positions_list_cur = [item+base for item in positions_list]
        bplot = plt.boxplot(data, sym="+", patch_artist=True, labels=methods_List, positions=positions_list_cur,widths=widths,flierprops={'color':'black','markeredgecolor':"black"}) 
        #将三个箱分别上色
        for patch, median, color in zip(bplot['boxes'], bplot['medians'], module_color_list):
            median.set(color=color, linewidth=1)
            patch.set(color=color, linewidth=1)
            patch.set_facecolor("white")

        for whisker,cap,color in zip(bplot['whiskers'],bplot['caps'], module_color_list2):
            whisker.set(color=color, linewidth=1, linestyle="--")
            cap.set(color=color, linewidth=1)

    plt.xticks([i + 0.8 / 2 for i in x_name_by_biology_position], x_name_by_biology_new,  fontsize=12 )
    plt.yticks(fontsize=12)
    # plt.legend(bplot['boxes'],methods_List,loc='lower right', bbox_to_anchor=(1, 1), ncol=len(methods_List), frameon=False,handlelength=1.4)  #绘制表示框，右下角绘制
    plt.legend(bplot['boxes'], methods_List, loc='lower right', bbox_to_anchor=(1, 1), ncol=1, frameon=False, handlelength=1.4)
    # plt.savefig(fname="/home/nellie/code/cvpr/Project/DynamiCrafter/main/DiceBrain.png",bbox_inches='tight',dpi=200)
    # plt.savefig(fname="/home/nellie/code/cvpr/Project/DeepLDDMM/2026ISBIExperiments/PaperViewRecords/OASIS/DiceBrain.png",bbox_inches='tight',dpi=200)
    plt.savefig(fname=saveP,bbox_inches='tight',dpi=200)
    plt.show()

