from functools import partial
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Optional

import os, sys
# print(sys.path[0])  #/home/bsw3ac/.conda/envs/wn/lib/python311.zip   /sfs/weka/scratch/bsw3ac/nellie/code/cvpr/Project/LightningTemplate/scripts

# print("\n\n")
# assert 4>10

# os.chdir("/home/nellie/code/cvpr/Project/LightningTemplate/scripts")
# sys.path.append("..")


from utils.utils import instantiate_from_config

def collate_stack_or_list(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    imgs = [b["series"] for b in batch]
    paths = [b["path"] for b in batch]
    origs = [b["orig_shape"] for b in batch]
    lv_segs = [b["lv_seg"] for b in batch if b["lv_seg"] is not None]
    index_endEstolic = [b["index_endEstolic"] for b in batch if b["index_endEstolic"] is not None]
    index_endEstolic2 = [b["index_endEstolic2"] for b in batch if b["index_endEstolic2"] is not None]
    idx = [b["idx"] for b in batch]
    vol_endEstolic = [b["vol_endEstolic"] for b in batch]
    lv_seg_endEstolic = [b["lv_seg_endEstolic"] for b in batch if b["lv_seg_endEstolic"] is not None]

    # Try to stack if shapes match
    same_shape = all(tuple(imgs[0].shape) == tuple(x.shape) for x in imgs)
    if same_shape:
        images = torch.stack(imgs, dim=0)  # (B, T, W, H)
        lv_segs = torch.stack(lv_segs, dim=0) if lv_segs else None

    else:
        images = imgs  # a list of tensors with variable shapes
        lv_segs = lv_segs if lv_segs else None
   

    return {"series": images, "paths": paths, "orig_shapes": origs, "lv_segs": lv_segs, "idx": idx, "index_endEstolic2": index_endEstolic2, "index_endEstolic": index_endEstolic, "vol_endEstolic":vol_endEstolic, "lv_seg_endEstolic":lv_seg_endEstolic}




class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, batch_size_test=20, train=None, valid=None, test=None, num_workers=None, *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        

        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        # print(self.num_workers)
        # assert 4>444
       
        self.Trainshuffle = kwargs['Trainshuffle'] if 'Trainshuffle' in kwargs else True
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if valid is not None:
            self.dataset_configs["valid"] = valid
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)


    def _train_dataloader(self):

        return DataLoader(self.datasets["train"], batch_size=self.batch_size, pin_memory=True, persistent_workers=(self.num_workers > 0),
                          num_workers=self.num_workers, shuffle=self.Trainshuffle, collate_fn=collate_stack_or_list, drop_last=False)

    def _val_dataloader(self):
        return DataLoader(self.datasets["valid"], batch_size=self.batch_size, pin_memory=True, persistent_workers=(self.num_workers > 0),
                          num_workers=self.num_workers, shuffle=False, collate_fn=collate_stack_or_list, drop_last=False)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size_test, pin_memory=True, persistent_workers=(self.num_workers > 0),
                          num_workers=self.num_workers, shuffle=False, collate_fn=collate_stack_or_list, drop_last=False)


