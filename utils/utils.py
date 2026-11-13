import importlib
import numpy as np
import cv2
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F



class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
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

    def forward(self, src, flow, mode=None):
        if mode is None:
            mode = self.mode
        # new locations
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

        warped = F.grid_sample(src, new_locs, mode=mode)
        return (warped, new_locs_unnormalize)









class Svf(nn.Module):
    def __init__(self, inshape, steps=7):
        super().__init__()
        self.nsteps = steps
        assert self.nsteps >= 0, 'nsteps should be >= 0, found: %d' % self.nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
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
        # print(dispList[-1].shape)  #torch.Size([24, 2, 64, 64])
        # assert 3>999



        u = vec

        
        grid_yx_pix = self.transformer.grid + u            # [B,2,H,W] (y,x) pixel
        # normalize to [-1,1] using (s-1)  => align_corners=True
        H, W = u.shape[2], u.shape[3]
        grid_yx_norm = grid_yx_pix.clone()
        grid_yx_norm[:,0] = 2*(grid_yx_pix[:,0]/(H-1) - 0.5)    # y
        grid_yx_norm[:,1] = 2*(grid_yx_pix[:,1]/(W-1) - 0.5)    # x

        # to grid_sample format (x,y)
        G_def = grid_yx_norm.permute(0,2,3,1)[..., [1,0]]       # [B,H,W,2] (x,y)

        return vec, dispList, G_def   #len





# def dice_coefficient(array_a, array_b):
#     intersection = np.sum(array_a * array_b)  # 元素相乘后求和
#     union = np.sum(array_a) + np.sum(array_b)
#     if union == 0:
#         return 1.0  # 如果两者都为 0，返回 1.0
#     return (2.0 * intersection) / union



def dice_coefficient(tensor_a, tensor_b, return_mean=True):
    # #print("dice_tensor_shape:  ",tensor_a.shape, tensor_b.shape)    #[10, 1, 128, 128] [10, 1, 128, 128]
    tensor_a = tensor_a.clone()
    tensor_b = tensor_b.clone()
    tensor_a[tensor_a<0.5] = 0
    tensor_a[tensor_a>=0.5] = 1
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
    
    union[union==0] = 0.0000000001
    res = (2.0 * intersection) / union

    if return_mean:
        return torch.mean(res).item()
    res = [item.item() for item in res]
    return res





def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """ 
    name: full name of source para
    para_list: partial name of target para 
    """
    istarget=False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    data = [np.load(os.path.join(data_dir, data_name))['arr_0'] for data_name in os.listdir(data_dir)]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)['arr_0'] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data   


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )

import yaml
import copy

def recursive_merge(dct, merge_dct):
    for k, v in merge_dct.items():
        if (
            k in dct and isinstance(dct[k], dict)
            and isinstance(merge_dct[k], dict)
        ):
            recursive_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

def merge_yaml_files(base_file, edit_file, output_file):
    """
    递归合并两个 YAML 文件的内容。
    
    :param base_file: 基础 YAML 文件路径
    :param edit_file: 编辑 YAML 文件路径
    :param output_file: 输出合并后的 YAML 文件路径
    """
    with open(base_file, "r") as f:
        base_config = yaml.safe_load(f)

    with open(edit_file, "r") as f:
        edit_config = yaml.safe_load(f)

    # 做深拷贝，避免改动原始
    merged_config = copy.deepcopy(base_config)

    # 递归合并
    recursive_merge(merged_config, edit_config)

    # 输出结果
    with open(output_file, "w") as f:
        yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)

    #print(f"✅ Merged configuration has been saved to {output_file}")









    

def main():
    # 读取basic.yaml
    with open("basic.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    # 读取edit.yaml
    with open("edit.yaml", "r") as f:
        edit_config = yaml.safe_load(f)

    # 做深拷贝，避免改动原始
    merged_config = copy.deepcopy(base_config)

    # 递归合并
    recursive_merge(merged_config, edit_config)

    # 输出结果
    with open("merged.yaml", "w") as f:
        yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)

    #print("✅ Merged configuration has been saved to merged.yaml")
    

if __name__ == "__main__":
    main()
