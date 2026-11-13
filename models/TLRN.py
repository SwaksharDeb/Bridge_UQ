import torch
import numpy as np
import torch.nn as nn
from utils.utils import Svf
from utils.Int import plot_dfm_process, plot_dfm_process_for_series, dice_coefficient, compute_dice_det
from utils.Int import jacobian_det_for_one, dice_coefficient_tensor, VecInt, SpatialTransformer, jacobian_det_for_all, save_dfm_dfmlabel, get_jacobian_det_for_all, jacobian_determinant, \
    Epdiff, Grad, Mgridplot, add_zero_channel_and_norm, plot_norm_list_gpu, dice_coefficient_for_brain, hdorff_coefficient_for_brain, dice_coefficient_for_cardiac
import pytorch_lightning as pl
import time
from pathlib import Path
import re

from utils.utils import instantiate_from_config
from torchmetrics import Metric
import torchmetrics


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, kernal=3, stride=1, padding=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernal, stride, padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock_github(nn.Module):  ## resnetblock
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_github, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.2)

        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = None

        self.stride = stride
    def forward(self, x):   #[4, 32, 8, 8]
        identity = x

        out = self.conv1(identity)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
        # return identity   

class BasicBlock_ablation(nn.Module):  ## resnetblock
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_github, self).__init__()
             
    def forward(self, x):   #[4, 32, 8, 8]
        b,c,h,w = x.shape
        return x[:,:int(c/2),...] + x[:,-int(c/2):,...]  #





class SNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SNConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        return self.norm(x)

class Net2DResNet(pl.LightningModule):
    def __init__(self, reg_config, regis_lr, weight_decay, weight_reg, inshape, weight_mse,
                 *args, **kwargs):
        super(Net2DResNet, self).__init__()
        self.workdir = kwargs['workdir'] if 'workdir' in kwargs else "./"
        self.test_split = kwargs['test_split'] if 'test_split' in kwargs else "test"
        self.Start_Subject_ID = kwargs['Start_Subject_ID'] if 'Start_Subject_ID' in kwargs else "0"
        self.target_T = kwargs['target_T'] if 'target_T' in kwargs else 15
        self.editfile = kwargs['editfile'] if 'editfile' in kwargs else "None"
        self.ch_sz = kwargs['ch_sz'] if 'ch_sz' in kwargs else 2

        
        self.regmetric = instantiate_from_config(reg_config)

        self.regis_lr = regis_lr
        self.weight_decay = weight_decay

        self.weight_mse =  weight_mse
        if 'weight_mse_pow' in kwargs:
            [a, r] = kwargs['weight_mse_pow']
            vals = [a * (r ** k) for k in range(self.target_T-1)]
            # weights = torch.tensor(vals)
            weights = torch.tensor(vals, device=self.device)
            # print(self.device)
            # print("\n\n\n")
            # assert 4>455
            """ print(weights)
            tensor([1.0000, 1.1000, 1.2100, 1.3310, 1.4641, 1.6105, 1.7716, 1.9487, 2.1436, 2.3579, 2.5937, 2.8531, 3.1384, 3.4523, 3.7975]) """
            weight_mse_list = weights / weights.sum()  # normalize (optional)
            
            self.register_buffer("weight_mse_list", weight_mse_list, persistent=False)
            # self.weight_mse_list = 

            # print(self.weight_mse_list)
            # self.mse_metric = nn.MSELoss(reduction='none')
            if 'MSE_config' in kwargs:
                kwargs['MSE_config']['params']['reduction'] = 'mean'
                self.mse_metric = instantiate_from_config(kwargs['MSE_config'])
            else:
                self.mse_metric = nn.MSELoss(reduction='none')

            self.compute_loss = self.compute_loss_Vary_MSE_W
        else:
            self.weight_mse_list = None
            # self.mse_metric = nn.MSELoss()
            if 'MSE_config' in kwargs:
                kwargs['MSE_config']['params']['reduction'] = 'mean'
                self.mse_metric = instantiate_from_config(kwargs['MSE_config'])
            else:
                self.mse_metric = nn.MSELoss()
            
            self.compute_loss = self.compute_loss_Fix_MSE_W

        # assert 4>444


        self.weight_reg = weight_reg
        

        self.args = args
        self.inshape = inshape
        self.dice_list = []
        self.hd95_list = []
        self.det_list = []
        self.dice_list_endEstolic = []
        self.idx_list_endEstolic = []
        self.sim_list = []
        self.sim_list_masked = []
        self.sim_list_masked_2 = []

        self.series_len = 15
        self.one_residue_block = False
        self.residue_block_num = (self.series_len-2) if not self.one_residue_block else 1
        self.reslevel = 3

        ch_sz = self.ch_sz
        nb_features=[[16*ch_sz, 32*ch_sz, 32*ch_sz], [32*ch_sz, 32*ch_sz, 32*ch_sz, 32*ch_sz, 16*ch_sz, 16]]


        infeats=2
        nb_levels=None
        max_pool=2
        feat_mult=1
        nb_conv_per_level=1
        half_res=False
        
        self.TSteps = 7
        self.img_size = inshape[0] if hasattr(inshape, '__getitem__') else 64
        self.test_img_size = self.img_size
        self.ablation_no_resnet = False
        self.one_residue_block = False
        # self.id = torch.from_numpy(lm.identity((1,2,args.img_size,args.img_size))).cuda()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        self.final_nf = prev_nf


        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.final_nf, ndims, kernel_size=3, padding=1)
        
        for idex in range(len(self.encoder)):
            self.encoder[idex][0].main.name = f'unet-encoder-{idex}-0'
            self.decoder[idex][0].main.name = f'unet-decoder-{idex}-0'
            
        for idex in range(len(self.remaining)):
            self.remaining[idex].main.name = f'unet-remaining-{idex}'
        


        self.flow.name = 'unet-flow'
        self.residue_block = nn.ModuleList()
        for i in range(self.residue_block_num):
            OneResBlock = nn.ModuleList()
            for level in range(self.reslevel):
                if (level==0):
                    # OneResBlock.append(BasicBlock_github(inplanes=32, planes=32))
                    OneResBlock.append(BasicBlock_github(inplanes=ch_sz*32*2, planes=ch_sz*32))
                else:
                    OneResBlock.append(BasicBlock_github(inplanes=ch_sz*32, planes=ch_sz*32))
            self.residue_block.append(OneResBlock)
            self.residue_block[i].name = f'residual-block-{i}'



        self.MSvf = Svf(inshape=(self.img_size, self.img_size), steps=self.TSteps)





    
    def exec_encoder(self, x):
        # print(x.shape) #torch.Size([12, 2, 64, 64])
        if(x.shape[-1]==2):
            x = x.permute(0,3,1,2)
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)  #100x100 -> 50x50 -> 25x25  ->12x12  ->6x6
                                        #128x128   64x64   32x32   16x16
            # print(level, "~~~~~~~~~~~~", x.shape, "\n\n")
            # 0 ~~~~~~~~~~~~ torch.Size([12, 16, 32, 32]) 
            # 1 ~~~~~~~~~~~~ torch.Size([12, 32, 16, 16]) 
            # 2 ~~~~~~~~~~~~ torch.Size([12, 32, 8, 8]) 
        return x, x_history

    def exec_decoder(self, x, x_history):
        skip_index = -1
        # print(len(self.decoder))
        
        for level, convs in enumerate(self.decoder):
            # print(level, "1-level ~~~~~~~~~~~~~~~", "\n")
            for conv in convs:
                x = conv(x)
            # print(level, "2-level ~~~~~~~~~~~~~~~", "\n")
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                # #print("x.shape: ", x.shape, "  x_history[skip_index].shape: ", x_history[skip_index].shape)
                x = torch.cat([x, x_history[skip_index]], dim=1)
                skip_index -= 1
            # print(level, "3-level ~~~~~~~~~~~~~~~", "\n")

        # print(len(self.remaining))
        for conv in self.remaining:
            x = conv(x)
            # print(level, "4-level ~~~~~~~~~~~~~~~", "\n")

        # print("5-level ~~~~~~~~~~~~~~~", "\n")
        # print(self.flow)
        # print(x.shape)
        x = self.flow(x)  #[20, 16, 64, 64] -> [20, 2, 64, 64]
        # assert 2>987

        if(len(x.shape)==4):
            return x.permute(0,2,3,1)   #-> [20, 64, 64, 2]
        return x.permute(0,2,3,4,1)   #-> [20, 64, 64, 64,3]

    def pair_register(self, x, masks=None):
        self.src = x[:,0:1,...] #[32, 1, 64, 64]
        self.tar = x[:,1:2,...] #[32, 1, 64, 64]
        if masks is not None:
            self.src_mask = masks[:,0:1,...]

        low_dim_features,x_his = self.exec_encoder(x)  #[b, 32, 8, 8]
        fnow_full_dim = self.exec_decoder(low_dim_features,x_his) #[b, 64, 64, 2]
        v = torch.permute(fnow_full_dim, [0, 3, 1, 2]) #[b, 64, 64, 2] -> [b, 2, 64, 64]

        u, u_seq, _ = self.MSvf(v) #u:[42, 2, 128, 128]
        
        Sdef,phiinv = self.MSvf.transformer(self.src, u)   #sdef: [32, 1, 64, 64]  phiinv:[42, 128, 128, 2]
       
        if masks is not None:
            Sdef_mask, _ = self.MSvf.transformer(self.src_mask, u, mode="nearest")
        else:
            Sdef_mask = None
        
        u_inv, _, _ = self.MSvf(-v)
        _,phi = self.MSvf.transformer(self.src, u_inv)

        return Sdef, v, [phiinv], Sdef_mask, [phi]
   
    def sequence_register_no_avg_lowf_addlatentf(self, input, masks=None):  #input   series [4, 9, 64, 64]  [32, 3, 128, 128]
        b, seq_num, h, w = input.shape
        self.src = input[:,0:1,...]     #[4, 1, 64, 64]
        if masks is not None:
            self.src_mask = masks[:,0:1,...]
        low_dim_features_seq = []; x_history_seq = []; low_dim_features_merge_seq = []
        v_series = []; Sdef_series = []; u_series = []; Sdef_mask_series = []; ui_series = []
        for cnt  in range(1, seq_num):
            self.second = input[:,cnt:cnt+1,...]  #[4, 1, 64, 64]
            firt_input = torch.cat((self.src, self.second), dim=1)
            low_dim_features, x_his = self.exec_encoder(firt_input)  #[b, 32, 8, 8]
            low_dim_features_seq.append(low_dim_features)  #length: 8
            x_history_seq.append(x_his)
        low_dim_features_merge_seq = []
        low_dim_features1 = low_dim_features_seq[0]
        low_dim_features_merge_seq.append(low_dim_features1*1.0)
        for cnt in range(1, len(low_dim_features_seq)):
            low_dim_features2 = low_dim_features_seq[cnt]
            inputX = torch.cat((low_dim_features1, low_dim_features2), dim=1)
            if self.ablation_no_resnet:
                inputX =  (low_dim_features1 + low_dim_features2) / 2 
            elif self.one_residue_block:
                for level, conv in enumerate(self.residue_block[0]):
                    inputX = conv(inputX)
            else:
                for level, conv in enumerate(self.residue_block[cnt-1]):
                    inputX = conv(inputX)
            low_dim_features_merge_seq.append(inputX)
            low_dim_features1 = inputX
        for cnt in range(0, seq_num-1):
            full_dim_features = self.exec_decoder(low_dim_features_merge_seq[cnt], x_history_seq[cnt])  #[4, 64, 64, 2]
            full_dim_features = full_dim_features.permute(0,3,1,2)  #[4, 2, 64, 64]
            v_series.append(full_dim_features)
            u,_,_ = self.MSvf(full_dim_features)
            Sdef, phiinv = self.MSvf.transformer(self.src, u)    #[4, 1, 64, 64]
            ui,_,_ = self.MSvf(-full_dim_features)
            _, phi = self.MSvf.transformer(self.src, ui)   #[4, 1, 64, 64]
            if masks is not None:
                Sdef_mask, _ = self.MSvf.transformer(self.src_mask, u, mode="nearest")
                Sdef_mask_series.append(Sdef_mask)
            Sdef_series.append(Sdef)
            u_series.append(u)   #phiinv 对应的disp
            ui_series.append(ui) #phi 对应的disp
        if masks is None:
            Sdef_mask_series = None
        return Sdef_series, v_series, u_series, Sdef_mask_series, ui_series





    def compute_loss_Vary_MSE_W(self, input, Sdef_series, v_series, u_series, ui_series, Sdef_mask_series=None):  #input: [4, 9, 64, 64]
        Tars = input[:, 1:, :,:].contiguous()
        Sdefs = torch.cat(Sdef_series, dim=1).contiguous()
        Velocities = torch.stack(v_series, dim=1)  ## [B,T,C,H,W] or [B,T,C,D,H,W]
        # print(Tars.shape, Sdefs.shape, Velocities.shape)
        #torch.Size([2, 14, 64, 64]) torch.Size([2, 14, 64, 64]) torch.Size([2, 14, 2, 64, 64])

        loss_per_frame  = self.mse_metric(Sdefs, Tars).mean(dim=(2, 3))   #torch.Size([b, 14]) 
        weighted_loss = (loss_per_frame * self.weight_mse_list).sum(dim=1).mean()
        loss1_series = weighted_loss
        loss2_series =  self.regmetric(Velocities)
        loss_series = loss1_series + self.weight_reg*loss2_series
        
        dice_series = 0
        for i in range(len(Sdef_series)):
            if Sdef_mask_series is not None:
                Sdef_mask = Sdef_mask_series[i]   #1,64,64
                tar_mask = input[:,i+1:i+2,:,:]   #1,64,64
                dice = dice_coefficient_tensor(Sdef_mask, tar_mask, return_mean=True)
            else:
                dice = 0
            dice_series += dice
        dice_series /= len(Sdef_series)

        return loss_series, loss1_series, loss2_series, dice_series
    


    def compute_loss_Fix_MSE_W(self, input, Sdef_series, v_series, u_series, ui_series, Sdef_mask_series=None):  #input: [4, 9, 64, 64]
        Tars = input[:, 1:, :,:].contiguous()
        Sdefs = torch.cat(Sdef_series, dim=1).contiguous()
        Velocities = torch.stack(v_series, dim=1)  ## [B,T,C,H,W] or [B,T,C,D,H,W]
        # print(Tars.shape, Sdefs.shape, Velocities.shape)
        #torch.Size([2, 14, 64, 64]) torch.Size([2, 14, 64, 64]) torch.Size([2, 14, 2, 64, 64])

        loss1_series = self.mse_metric(Tars, Sdefs)
        loss2_series =  self.regmetric(Velocities)
        loss_series = self.weight_mse*loss1_series + self.weight_reg*loss2_series

        dice_series = 0
        for i in range(len(Sdef_series)):
            if Sdef_mask_series is not None:
                Sdef_mask = Sdef_mask_series[i]   #1,64,64
                tar_mask = input[:,i+1:i+2,:,:]   #1,64,64
                dice = dice_coefficient_tensor(Sdef_mask, tar_mask, return_mean=True)
            else:
                dice = 0
            dice_series += dice
        dice_series /= len(Sdef_series)

        return loss_series, loss1_series, loss2_series, dice_series
    


    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        if self.epoch_start_time is not None:
            elapsed_time = time.time() - self.epoch_start_time
            print(f"\n\n TLRN_{self.editfile}  Epoch {self.current_epoch} ended. Time elapsed: {elapsed_time:.2f} seconds. \n\n")

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        slices_series = batch['series']  #series
        lv_seg = batch['lv_segs']        #lv_seg
        bs, seq_num, h, w = slices_series.shape
        # print(bs, "\n\n\n")


        # Sdef_series, v_series, u_series, ui_series, Sdef_mask_series = self.sequence_register_no_avg_lowf_addlatentf(slices_series, masks=lv_seg)
        Sdef_series, v_series, u_series, Sdef_mask_series, ui_series = self.sequence_register_no_avg_lowf_addlatentf(slices_series, masks=lv_seg)
        # u_series.append(u)   #phiinv 对应的disp
        # ui_series.append(ui) #phi 对应的disp
        
        loss_series, loss1_series, loss2_series, dice_series = self.compute_loss(slices_series, Sdef_series, v_series, u_series, ui_series, Sdef_mask_series)

        # print(loss_series, loss1_series, loss2_series)
        # assert 3>444

        self.log_dict(
            {
                "train_loss": loss_series,
                "train_mse": loss1_series,
                "train_reg": loss2_series,
            },
            batch_size=bs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss_series

    
    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        slices_series = batch['series']  #series
        lv_seg = batch['lv_segs']        #lv_seg
        bs, seq_num, h, w = slices_series.shape
        # print(bs, "\n\n\n")

        # Sdef_series, v_series, u_series, ui_series, Sdef_mask_series = self.sequence_register_no_avg_lowf_addlatentf(slices_series, masks=lv_seg)
        Sdef_series, v_series, u_series, Sdef_mask_series, ui_series = self.sequence_register_no_avg_lowf_addlatentf(slices_series, masks=lv_seg)
        # u_series.append(u)   #phiinv 对应的disp
        # ui_series.append(ui) #phi 对应的disp

        loss_series, loss1_series, loss2_series, dice_series = self.compute_loss(slices_series, Sdef_series, v_series, u_series, ui_series, Sdef_mask_series)

        self.log_dict(
            {
                "val_loss": loss_series,
                "val_mse": loss1_series,
                "val_reg": loss2_series,
            },
            batch_size=bs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss_series


    def largest_def_and_score(self, TA: torch.Tensor, TB: torch.Tensor):
        """
        Returns:
        idx_max (N,)      -> index of the largest L2-norm deformation per example
        dice_at_max (N,)  -> dice score at that index per example
        """
        assert TA.dim() == 5 and TB.dim() == 2, "Shapes must be TA=(N,T,2,H,W), TB=(N,T)"
        N, T, C, H, W = TA.shape
        assert C == 2 and TB.shape == (N, T), "Channel must be 2; TB must match (N,T)."
        # L2 norm of each deformation (over channels and spatial dims): (N, T)
        l2_per_frame = torch.sqrt((TA ** 2).sum(dim=(2, 3, 4)))
        # Index of the largest deformation for each example: (N,)
        idx_max = torch.argmax(l2_per_frame, dim=1)
        idx_max = idx_max.cpu()
        # print(idx_max)
        # tensor([ 9,  9,  6,  8,  3,  9,  9, 10, 10,  7, 10, 10,  6,  9,  8,  9, 10,  9, 9, 12])
        # assert 4>5555
        # Gather the corresponding dice score from TB: (N,)
        dice_at_max = TB.gather(1, idx_max.unsqueeze(1)).squeeze(1)
        return idx_max, dice_at_max

   
    def find_dice_for_endEstolic(self, u_series, dice_list):
        all_u_series_ = torch.stack(u_series, dim=1)  #[20, 14, 2, 64, 64]
        # print(all_u_series_.shape, type(all_u_series_)) 
        # print(len(dice_list), dice_list) #20
        dice_list_tensor = [torch.tensor(dice) for dice in dice_list]
        all_dice = torch.stack(dice_list_tensor, dim=0)  #[20, 14]
        # print(all_dice.shape, type(all_dice), "\n\n")
        idx, dice = self.largest_def_and_score(all_u_series_, all_dice)
        dice_list_endEstolic = [item.item() for item in dice]
        idx_list_endEstolic = [item.item() for item in idx]
        return idx_list_endEstolic, dice_list_endEstolic


    def find_dice_hd95_for_endEstolic_with_index(self, index_endEstolic, dice_list, hd95_list): 
        # index_endEstolic: list/iterable of scalar tensors (end-systolic frame indices)
        # dice_list: list of lists, each inner list = dice over frames for one sample
        # hd95_list: same structure as dice_list, but HD95 values

        # Convert indices to a tensor
        idx_max = torch.tensor([item.item() for item in index_endEstolic])  # shape [B]

        # ---------- DICE ----------
        # Pad each sequence with a leading 0 so index 0 is "dummy"
        dice_list = [[0] + item for item in dice_list]
        dice_list_tensor = [torch.tensor(d) for d in dice_list]
        TB = torch.stack(dice_list_tensor, dim=0)  # [B, Frame]

        B, Frame = TB.shape
        dice_at_max_list = []
        for i in range(B):
            idx = int(idx_max[i])
            if idx < Frame:
                dice_at_max_list.append(TB[i, idx])
            else:
                dice_at_max_list.append(torch.tensor(0.0, dtype=TB.dtype))

        dice_at_max = torch.tensor(dice_at_max_list)  # shape [B]
        dice_list_endEstolic = [item.item() for item in dice_at_max]

        # ---------- HD95 ----------
        # Same treatment for HD95
        hd95_list = [[0] + item for item in hd95_list]
        hd95_list_tensor = [torch.tensor(h) for h in hd95_list]
        HB = torch.stack(hd95_list_tensor, dim=0)  # [B, Frame]

        _, Frame_hd = HB.shape
        hd95_at_max_list = []
        for i in range(B):
            idx = int(idx_max[i])
            if idx < Frame_hd:
                hd95_at_max_list.append(HB[i, idx])
            else:
                hd95_at_max_list.append(torch.tensor(0.0, dtype=HB.dtype))

        hd95_at_max = torch.tensor(hd95_at_max_list)  # shape [B]
        hd95_list_endEstolic = [item.item() for item in hd95_at_max]

        # Indices as plain Python ints
        idx_list_endEstolic = [int(item.item()) for item in idx_max]

        return idx_list_endEstolic, dice_list_endEstolic, hd95_list_endEstolic




    def find_dice_for_endEstolic_with_index(self, index_endEstolic, dice_list, hd95_list):
        idx_max = torch.tensor([item.item() for item in index_endEstolic])
        dice_list = [[0]+item for item in dice_list]
        dice_list_tensor = [torch.tensor(dice) for dice in dice_list]
        TB = torch.stack(dice_list_tensor, dim=0)  #[20, 14]
        # dice_at_max = TB.gather(1, idx_max.unsqueeze(1)).squeeze(1)
        dice_at_max_list = []
        B, Frame = TB.shape
        for i in range(B):
            if idx_max[i]<Frame:
                dice_at_max_list.append(TB[i, idx_max[i]])
            else:
                dice_at_max_list.append(0)
        dice_at_max = torch.tensor(dice_at_max_list)  # shape: [B]

        dice_list_endEstolic = [item.item() for item in dice_at_max]
        idx_list_endEstolic = [item.item() for item in idx_max]
        return idx_list_endEstolic, dice_list_endEstolic
    

    def collect_nonzero_values(self, list_of_lists):
        result_list = []
        index_list = []
        for list_idx, lst in enumerate(list_of_lists):
            for elem_idx, x in enumerate(lst):
                if x != 0:
                    result_list.append(x)
                    index_list.append(elem_idx)
                    break
        return index_list, result_list
    
    
    def save_npyfile_for_gif(self, Sdef_series_all, rawframes_series_list, ui_series_all, paths):
        batches = ui_series_all[0].shape[0]
        save_folder = f"{self.workdir}"
        
        for i_batch in range(batches):
            old = Path(paths[i_batch])

            ui_series = [item[i_batch: i_batch+1] for item in ui_series_all]
            dfm_series = [item[i_batch: i_batch+1] for item in Sdef_series_all]
            rawframe_series =  [item[i_batch: i_batch+1] for item in rawframes_series_list]
            
            #displacement save for Rishov
            ui_series_rishov = torch.cat(ui_series, dim=0)
            dfm_series_rishov = torch.cat(dfm_series, dim=0)
            rawframe_series_rishov = torch.cat(rawframe_series, dim=0)

            # ui_series_rishov = ui_series_rishov.permute(3,0,1,2).unsqueeze(0)
            # print(ui_series_rishov.shape) #[14, 2, 64, 64]  #1, 2, f, h, w = displacement shape
            ui_series_rishov = ui_series_rishov.permute(1,0,2,3).unsqueeze(0)
            dfm_series_rishov = dfm_series_rishov.permute(1,0,2,3).unsqueeze(0)
            rawframe_series_rishov = rawframe_series_rishov.permute(1,0,2,3).unsqueeze(0)

            print(ui_series_rishov.shape, dfm_series_rishov.shape, rawframe_series_rishov.shape)
            # print(old)
            # assert 4>987
            #nii.gz path name
            new = Path(save_folder) / f"Miccai2026JanGIF/displacement_{self.test_split}" / old.name
            new.parent.mkdir(parents=True, exist_ok=True)
            #.npy path name
            new_name = re.sub(r'\.nii(\.gz)?$', '.npy', old.name)
            new_path = Path(save_folder) / f"Miccai2026JanGIF/displacement_{self.test_split}" / new_name
            np.save(new_path, ui_series_rishov.detach().cpu().numpy())
            # print(i_batch, ui_series_rishov.shape, new_path)

            new = Path(save_folder) / f"Miccai2026JanGIF/deforms_{self.test_split}" / old.name
            new.parent.mkdir(parents=True, exist_ok=True)
            new_path = Path(save_folder) / f"Miccai2026JanGIF/deforms_{self.test_split}" / new_name
            np.save(new_path, dfm_series_rishov.detach().cpu().numpy())
            # print(i_batch, dfm_series_rishov.shape, new_path)

            new = Path(save_folder) / f"Miccai2026JanGIF/rawframes_{self.test_split}" / old.name
            new.parent.mkdir(parents=True, exist_ok=True)
            new_path = Path(save_folder) / f"Miccai2026JanGIF/rawframes_{self.test_split}" / new_name
            np.save(new_path, rawframe_series_rishov.detach().cpu().numpy())


    def test_step_git(self, batch, batch_idx, optimizer_idx=None):
        slices_series = batch['series']  #series  #torch.Size([20, 15, 64, 64]) 
        slices_series_list = torch.split(slices_series, 1, dim=1)   
        lv_seg = batch['lv_segs']        #lv_seg
        paths = batch['paths']
        index_endEstolic = batch['index_endEstolic']
        Sdef_series, v_series, u_series, Sdef_mask_series, ui_series = self.sequence_register_no_avg_lowf_addlatentf(slices_series, masks=lv_seg)
        self.save_npyfile_for_gif([slices_series_list[0]] + Sdef_series, slices_series_list, ui_series, paths)
        # dice_list, det_list = compute_dice_det(slices_series, Sdef_series, u_series, ui_series, Sdef_mask_series, lv_seg, batch_idx, inshape=self.inshape)
        # idx_list_endEstolic, dice_list_endEstolic = self.find_dice_for_endEstolic_with_index(index_endEstolic, dice_list)
        # print("dice:  ", dice_list_endEstolic)


    def test_step(self, batch, batch_idx, optimizer_idx=None):
        slices_series = batch['series']  #series
        lv_seg = batch['lv_segs']        #lv_seg
        index_endEstolic = batch['index_endEstolic']

        Sdef_series, v_series, u_series, Sdef_mask_series, ui_series = self.sequence_register_no_avg_lowf_addlatentf(slices_series, masks=lv_seg)
        # u_series.append(u)   #phiinv 对应的disp
        # ui_series.append(ui) #phi 对应的disp
        dice_list, det_list, hd95_list, sim_list, _, sim_list_masked, sim_list_masked_2 = compute_dice_det(slices_series, Sdef_series, u_series, ui_series, Sdef_mask_series, lv_seg, batch_idx, inshape=self.inshape)
        idx_list_endEstolic, dice_list_endEstolic = self.find_dice_for_endEstolic_with_index(index_endEstolic, dice_list, hd95_list)
        
        self.dice_list += dice_list
        self.hd95_list += hd95_list

        self.det_list += det_list
        self.dice_list_endEstolic += dice_list_endEstolic
        self.idx_list_endEstolic += idx_list_endEstolic
        self.sim_list += sim_list
        self.sim_list_masked += sim_list_masked
        self.sim_list_masked_2 += sim_list_masked_2
        
        src = slices_series[:,0:1,...] #[N,1,64,64]
        srclabel = lv_seg[:,0:1,...] if lv_seg is not None else None #[N,1,64,64]

        return {
            "src": src,
            "tar_list": [slices_series[:,i:i+1,...] for i in range(1, slices_series.shape[1]) ],  #list of [N,1,64,64]
            "dfmSrc_list": Sdef_series,
            "VecF_List": v_series,

            "phiinv_disp_list": u_series,
            # "phi_disp_list": ui_series,

            "srclabel": srclabel,
            "tarlabel_list": [lv_seg[:,i:i+1,...] for i in range(1, lv_seg.shape[1]) ] if lv_seg is not None else None,  #list of [N,1,64,64]
            "dfmSrclabel_list": Sdef_mask_series,
        }


    def test_epoch_end11(self, outputs):
        save_folder = f"{self.workdir}"
        file3 = open(save_folder+f"/histmatch_Miccai2026Jan_{self.test_split}_dice.txt", "w")
        file6 = open(save_folder+f"/histmatch_Miccai2026Jan_{self.test_split}_dice_endEstolic.txt", "w")
        # write diff_list, dice_list, hausdorff_list, and det_list in to file through loop
        print(len(self.dice_list), len(self.det_list))
        for i in range(len(self.dice_list)):
            dice_str = str(self.dice_list[i]).replace("[", "").replace("]","").replace(","," ").replace("  "," ")
            file3.write(f"{dice_str}\n")
            
        for i in range(len(self.dice_list_endEstolic)):
            dice_endEstolic_str = str(self.dice_list_endEstolic[i]).replace("[", "").replace("]","").replace(","," ").replace("  "," ")
            file6.write(f"{dice_endEstolic_str}\n")
            
    def test_epoch_end____(self, outputs):
        # plot_dfm_process_for_series(outputs, savePath=f"{self.workdir}/test_{self.test_split}_dfm_process_{self.Start_Subject_ID}.png")
        # assert 3>888
        #open a file 
        save_folder = f"{self.workdir}"
        
        # file3 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_dice_sequence_register_consecutive_compose.txt", "w")
        # file6 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_dice_endEstolic_sequence_register_consecutive_compose.txt", "w")


        # file3 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_dice_sequence_register_consecutive.txt", "w")
        # file6 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_dice_endEstolic_sequence_register_consecutive.txt", "w")


        # file3 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_dice_sequence_register_consecutive_warpfirst2.txt", "w")
        # file6 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_dice_endEstolic_sequence_register_consecutive_warpfirst2.txt", "w")


        file4 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_hd95.txt", "w")
        # file3 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_dice_sequence_register_consecutive_velocity.txt", "w")
        # file6 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_dice_endEstolic_sequence_register_consecutive_velocity.txt", "w")
        
        # write diff_list, dice_list, hausdorff_list, and det_list in to file through loop
        print(len(self.dice_list), len(self.det_list))
        for i in range(len(self.dice_list)):
            # dice_str = str(self.dice_list[i]).replace("[", "").replace("]","").replace(","," ").replace("  "," ")
            # file3.write(f"{dice_str}\n")
            hd95_str = str(self.hd95_list[i]).replace("[", "").replace("]","").replace(","," ").replace("  "," ")
            file4.write(f"{hd95_str}\n")
            
        # for i in range(len(self.dice_list_endEstolic)):
        #     dice_endEstolic_str = str(self.dice_list_endEstolic[i]).replace("[", "").replace("]","").replace(","," ").replace("  "," ")
        #     file6.write(f"{dice_endEstolic_str}\n")

    def test_epoch_end__plot(self, outputs):
        plot_dfm_process_for_series(outputs, savePath=f"{self.workdir}/test_{self.test_split}_dfm_process_{self.Start_Subject_ID}.png")



    def test_epoch_end(self, outputs):
        #plot_dfm_process_for_series(outputs, savePath=f"{self.workdir}/test_{self.test_split}_dfm_process_{self.Start_Subject_ID}.png")
        # assert 3>888
        #open a file 
        save_folder = f"{self.workdir}"
        
        file3 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_dice.txt", "w")
        file4 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_hd95.txt", "w")

        
        file5 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_det.txt", "w")
        file6 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_dice_endEstolic.txt", "w")
        file7 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_index_endEstolic.txt", "w")

        file8 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_sim_list.txt", "w")
        file9 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_sim_list_masked.txt", "w")
        file10 = open(save_folder+f"/Miccai2026Jan_{self.test_split}_sim_list_masked_2.txt", "w")

        # write diff_list, dice_list, hausdorff_list, and det_list in to file through loop
        print(len(self.dice_list), len(self.det_list))
        for i in range(len(self.dice_list)):
            dice_str = str(self.dice_list[i]).replace("[", "").replace("]","").replace(","," ").replace("  "," ")
            file3.write(f"{dice_str}\n")

            sim_str = str(self.sim_list[i]).replace("[", "").replace("]","").replace(","," ").replace("  "," ")
            file8.write(f"{sim_str}\n")
            sim_masked_str = str(self.sim_list_masked[i]).replace("[", "").replace("]","").replace(","," ").replace("  "," ")
            file9.write(f"{sim_masked_str}\n")
            sim_masked_2_str = str(self.sim_list_masked_2[i]).replace("[", "").replace("]","").replace(","," ").replace("  "," ")
            file10.write(f"{sim_masked_2_str}\n")

            hd95_str = str(self.hd95_list[i]).replace("[", "").replace("]","").replace(","," ").replace("  "," ")
            file4.write(f"{hd95_str}\n")

            det_str = str(self.det_list[i]).replace("[", "").replace("]","").replace(","," ").replace("  "," ")
            file5.write(f"{det_str}\n")
            
        for i in range(len(self.idx_list_endEstolic)):
            dice_endEstolic_str = str(self.dice_list_endEstolic[i]).replace("[", "").replace("]","").replace(","," ").replace("  "," ")
            file6.write(f"{dice_endEstolic_str}\n")
            index_endEstolic_str = str(self.idx_list_endEstolic[i]).replace("[", "").replace("]","").replace(","," ").replace("  "," ")
            file7.write(f"{index_endEstolic_str}\n")





    def Mgridplot(self,u, ax, Nx=64, Ny=64, displacement=True, color='red', linewidth=0.5, IMG=None, **kwargs):
        """Given a displacement field, plot a displaced grid"""
        u = self.to_numpy(u)
        """Given a displacement field, plot a displaced grid"""
        

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
        # ax.set_xlim(ax.get_ylim())  # Swap x and y limits
        # ax.set_ylim(ax.get_xlim())  # Swap x and y limits
        ax.set_aspect('equal')
        ax.invert_yaxis()

    def configure_optimizers(self):
        optimizer_all = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.regis_lr, weight_decay=self.weight_decay
        )
        # optimizer_all = torch.optim.Adam(self.parameters(), lr=self.regis_lr, weight_decay=self.weight_decay)
        return optimizer_all