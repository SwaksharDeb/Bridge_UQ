# dataloader_nii.py
from pathlib import Path
from typing import Iterable, List, Optional
import os
import math
import argparse
import random
from typing import List, Tuple, Dict, Optional
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from skimage.exposure import match_histograms


# -----------------------------
# 0) Paths from your snippet
# -----------------------------
# Image data root. Paths in the TrainingList/TestingList txt files that are
# relative are resolved under it; absolute paths are used as they are.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DataFolder = os.environ.get("BRIDGEUQ_DATA_ROOT", str(_REPO_ROOT / "data"))
# Folder holding TrainingList/ and TestingList/ (default: this folder).
ListFolder = os.environ.get("BRIDGEUQ_LIST_DIR", str(Path(__file__).resolve().parent))

# All have labels: 90% train, 10% test
PPL_Path1  = f"{DataFolder}/In-House/UVA/Fred"
PPL_Path2  = f"{DataFolder}/In-House/UVA/Jonathan/Year1"
PPL_Path3  = f"{DataFolder}/In-House/UVA/Ken(Yu)-Multi-Institute/Unpaired"
PPL_Path4  = f"{DataFolder}/In-House/UVA/UVA-DENSE/Paired_CINE"
PPL_Path5  = f"{DataFolder}/In-House/UVA/UVA-DENSE/Unpaired_CINE"

# Two frames have labels: 100% train
PPL_Path6  = f"{DataFolder}/In-House/UVA/Nisha/Year1"
PPL_Path7  = f"{DataFolder}/Public_Data/ACDC"
PPL_Path8  = f"{DataFolder}/Public_Data/MICCAI_2020"

# One frame has labels: 100% train
PPL_Path9  = f"{DataFolder}/In-House/Minnesota/Derek/GEN"
PPL_Path10 = f"{DataFolder}/In-House/Minnesota/Derek/MYO"
PPL_Path11 = f"{DataFolder}/In-House/Minnesota/Derek/SARC"

PPL_List12 = f"{DataFolder}/Public_Data/SCD"
PPL_List12 = PPL_Path11


PPL_List13 = f"{DataFolder}/Dense/Rishov"

TrainPPL_List = [
    {"path": PPL_Path1,  "ratio": 0.9},
    {"path": PPL_Path2,  "ratio": 0.9},
    {"path": PPL_Path3,  "ratio": 0.9},
    {"path": PPL_Path4,  "ratio": 0.9},
    {"path": PPL_Path5,  "ratio": 0.9},
    {"path": PPL_Path6,  "ratio": 1.0},
    {"path": PPL_Path7,  "ratio": 1.0},
    {"path": PPL_Path8,  "ratio": 1.0},
    {"path": PPL_Path9,  "ratio": 1.0},
    {"path": PPL_Path10, "ratio": 1.0},
    {"path": PPL_Path11, "ratio": 1.0},
]

TestPPL_ID_List = [
    {"path": PPL_Path1, "ratio": 0.1},
    {"path": PPL_Path2, "ratio": 0.1},
    {"path": PPL_Path3, "ratio": 0.1},
    {"path": PPL_Path4, "ratio": 0.1},
    {"path": PPL_Path5, "ratio": 0.1},
]
TestPPL_OOD_List = [
    {"path": PPL_List13, "ratio": 1.0},
]



# -----------------------------
# 1) Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mat_and_expand(mat_path: str, target_Frames: int) -> np.ndarray:
    import scipy.io
    mat_contents = scipy.io.loadmat(mat_path)
    # 假设感兴趣的数据存储在名为 'data' 的变量中，根据实际情况修改
    if 'lv_seg' in mat_contents:
        data = mat_contents['lv_seg']
        phase1 = mat_contents['phase1'][0,0]
        phase2 = mat_contents['phase2'][0,0]

        target_data = np.zeros((target_Frames, data.shape[1], data.shape[2]), dtype=data.dtype)

        target_data[phase1-1] = data[0]
        if phase2<=target_Frames:
            target_data[phase2-1] = data[1]

    else:
        raise KeyError(f"'lv_seg' not found in the .mat file: {mat_path}")
    return target_data, phase2-1



def list_nii_files(root: str) -> List[str]:
    files = []
    if not os.path.isdir(root):
        #print(f"[WARN] Directory not found: {root}")
        return files
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            lower = fn.lower()
            if lower.endswith(".nii") or lower.endswith(".nii.gz"):
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files

def iter_write(path, strings, append=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"   # "w" = overwrite, "a" = append
    # buffering=1 => line-buffered; flushes on newline for text files
    with p.open(mode, encoding="utf-8", buffering=1) as f:
        for s in strings:
            f.write(f"{s}\n")
            f.flush()  # extra safety; okay to omit if buffering=1




def split_indices(n: int, train_ratio: float, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(math.floor(train_ratio * n))
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]
    return train_idx, test_idx

def build_splits_from_txt(
        split,
        train_txt,
        test_id_txt,
        test_ood_txt,
        test_id_lv_seg_txt,
        test_ood_lv_seg_txt,
        train_index_endEstolic_text, 
        test_id_index_endEstolic_text, 
        test_ood_index_endEstolic_text
):

    splits = load_splits_from_txt(
        split,
        train_txt = train_txt,
        test_id_txt = test_id_txt,
        test_ood_txt = test_ood_txt,
        test_id_lv_seg_txt = test_id_lv_seg_txt,
        test_ood_lv_seg_txt = test_ood_lv_seg_txt,
        train_index_endEstolic_text = train_index_endEstolic_text, 
        test_id_index_endEstolic_text = test_id_index_endEstolic_text, 
        test_ood_index_endEstolic_text = test_ood_index_endEstolic_text,

        root=Path(DataFolder),  # 如果 txt 里写的是相对路径，填根目录 Path("/scratch/...")；否则留空
        verify_paths=True,    # 打开以检查磁盘上是否存在
        verbose=True,
    )

    # 直接取用：
    train_files = splits['train_files']
    test_id_files = splits['test_id_files']
    test_ood_files = splits['test_ood_files']

    test_id_lv_seg_files = splits['test_id_lv_seg_files']
    test_ood_lv_seg_files = splits['test_ood_lv_seg_files']

    train_index_endEstolic_files = splits['train_index_endEstolic_files']
    test_id_index_endEstolic_files = splits['test_id_index_endEstolic_files']
    test_ood_index_endEstolic_files = splits['test_ood_index_endEstolic_files']

    return train_files, test_id_files, test_ood_files, test_id_lv_seg_files, test_ood_lv_seg_files, train_index_endEstolic_files, test_id_index_endEstolic_files, test_ood_index_endEstolic_files



# -----------------------------
# 2) Build file lists (train / test-ID / test-OOD)
# -----------------------------
def build_splits(
    train_cfg: List[Dict],
    test_id_cfg: List[Dict],
    test_ood_cfg: List[Dict],
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Ensure that for paths appearing in both train and test-ID,
    we split once and make them disjoint (e.g., 90/10).
    """
    rng = np.random.RandomState(seed)


    # Gather unique paths
    train_map   = {d["path"]: d["ratio"] for d in train_cfg}
    testid_map  = {d["path"]: d["ratio"] for d in test_id_cfg}
    testood_map = {d["path"]: d["ratio"] for d in test_ood_cfg}

    all_paths = set(train_map.keys()) | set(testid_map.keys() | set(testood_map.keys()))

    train_files: List[str] = []
    test_id_files: List[str] = []
    test_ood_files: List[str] = []

    for p in sorted(all_paths):
        files = list_nii_files(p)
        if len(files) == 0:
            #print(f"[INFO] No NIfTI files found under: {p}")
            continue

        if p in train_map and p in testid_map:
            # Split deterministically
            tr_idx, te_idx = split_indices(len(files), train_map[p], rng)
            for i in tr_idx:
                train_files.append(files[i])
            for i in te_idx:
                test_id_files.append(files[i])
        elif p in train_map:
            # All files to train (ratio is expected to be 1.0)
            if train_map[p] < 1.0:
                # If a partial ratio is configured but there is no test list, respect it
                n_keep = int(math.floor(train_map[p] * len(files)))
                chosen = rng.choice(len(files), size=n_keep, replace=False)
                train_files.extend([files[i] for i in chosen])
            else:
                train_files.extend(files)
        elif p in testid_map:
            # Only in the test-ID config: sample the given ratio
            n_keep = int(math.floor(testid_map[p] * len(files)))
            if n_keep == 0:
                n_keep = min(1, len(files))  # ensure at least one if possible
            chosen = rng.choice(len(files), size=n_keep, replace=False)
            test_id_files.extend([files[i] for i in chosen])
        elif p in testood_map:
            # Only in the test-OOD config: sample the given ratio
            n_keep = int(math.floor(testood_map[p] * len(files)))
            if n_keep == 0:
                n_keep = min(1, len(files))  # ensure at least one if possible
            chosen = rng.choice(len(files), size=n_keep, replace=False)
            test_ood_files.extend([files[i] for i in chosen])
    # test_ood_files = test_id_files

    test_id_files, test_id_lv_seg_files = screen_test_splits_with_lv_seg(test_id_files)
    test_ood_files, test_ood_lv_seg_files = screen_test_splits_with_lv_seg(test_ood_files)



    #print(f"[SPLIT] Train: {len(train_files)} | Test-ID: {len(test_id_files)} | Test-OOD: {len(test_ood_files)}")
    return train_files, test_id_files, test_ood_files, test_id_lv_seg_files, test_ood_lv_seg_files



def screen_test_splits_with_lv_seg(test_id_files):
    screen_result_raw = []
    screen_result_lv_seg = []
    for save_file in test_id_files:
        dst_lv_seg = Path(save_file).parent / "lv_seg" / Path(save_file).name   # insert 'label' before the file name
        if not dst_lv_seg.exists():
            print(f"[WARN] No LV_Seg found for test file: {save_file}")
        else:
            screen_result_raw.append(save_file)
            screen_result_lv_seg.append(str(dst_lv_seg))
    return screen_result_raw, screen_result_lv_seg
# -----------------------------
# 3) Dataset & collate
# -----------------------------
class NiftiDataset(Dataset):
    def __init__(
        self,
        split: str = "train",  # "train", "valid", or "test_id"  "test_ood_ACDC"  "test_ood_Rishov_Cine" "test_ood_Rishov_Dense"
        normalize: str = "minmax",  # "zscore", "minmax", or "none"
        resize: Optional[Tuple[int, int]] = None,  # (H, W)
        dtype: torch.dtype = torch.float32,
        FileLen: Optional[int] = None,
        drop_first_frame = 0,
        Start_Subject_ID = 0,
        dataBase: str = "0TLRN_Foundation",
        trainList: str = "train_files_wo_ACDC",
        train_index_endEstolic_list: str = "index_endEstolic_train_wo_ACDC_phi",
        

        testidList: str = "testid_files_frames24_25",
        # testid_files_frames24_25_NormPer
        testid_lvseg_List: str = "testid_lv_seg_files_frames24_25",
        testid_index_endEstolic_list: str = "index_endEstolic_test_id_frames24_25_phi",

       *args, **kwargs
    ):
        # train_files, test_id_files, test_ood_files, test_id_lv_seg_files, test_ood_lv_seg_files = build_splits(
        #     TrainPPL_List, TestPPL_ID_List, TestPPL_OOD_List, seed=42
        # )

        self.target_T = kwargs["target_T"] if "target_T" in kwargs else 15
        self.start_frame = kwargs["start_frame"] if "start_frame" in kwargs else "random"
        self.split = split
   



        TXT_Folder = ListFolder
        TXT_Folder_Rishov = f"{ListFolder}/TestingListRishov"

        train_txt_path = Path(TXT_Folder) / f"TrainingList/{trainList}.txt"
        train_index_endEstolic_path = Path(TXT_Folder) / f"TrainingList/{train_index_endEstolic_list}.txt" #Path(TXT_Folder) / "TrainingList/  .txt"


        test_id_txt_path = Path(TXT_Folder) / f"TestingList/{testidList}.txt"
        test_id_lv_seg_txt_path = Path(TXT_Folder) / f"TestingList/{testid_lvseg_List}.txt"
        test_id_index_endEstolic_path = Path(TXT_Folder) / f"TestingList/{testid_index_endEstolic_list}.txt" #Path(TXT_Folder) / "TestingList/  .txt"


        ### Set the path of OOD test files
        if split == "test_ood_ACDC":
            test_ood_txt_path = Path(TXT_Folder) / "TestingList/testood_ACDC_files.txt"
            test_ood_lv_seg_txt_path = Path(TXT_Folder) / "TestingList/testood_ACDC_lv_seg_files.txt"
            test_ood_index_endEstolic_path = Path(TXT_Folder) / "TestingList/index_endEstolic_test_ood_ACDC.txt"

        elif split == "test_ood_Rishov_Cine":
            test_ood_txt_path = Path(TXT_Folder_Rishov) / "test_cine_files.txt"
            test_ood_lv_seg_txt_path = Path(TXT_Folder_Rishov) / "test_cine_lv_seg_files.txt"
            test_ood_index_endEstolic_path = Path(TXT_Folder) / "TestingList/index_endEstolic_test_ood_ACDC.txt" #Path(TXT_Folder) / "TestingList/  .txt"

        elif split == "test_ood_Rishov_Dense":
            test_ood_txt_path = Path(TXT_Folder_Rishov) / "test_dense_files.txt"
            test_ood_lv_seg_txt_path = Path(TXT_Folder_Rishov) / "test_dense_lv_seg_files.txt"
            test_ood_index_endEstolic_path = Path(TXT_Folder) / "TestingList/index_endEstolic_test_ood_ACDC.txt" #Path(TXT_Folder) / "TestingList/  .txt"

        else:
            test_ood_txt_path = Path(TXT_Folder) / "TestingList/testood_files_frames24_25.txt"
            test_ood_lv_seg_txt_path = Path(TXT_Folder) / "TestingList/testood_lv_seg_files_frames24_25.txt"
            test_ood_index_endEstolic_path = Path(TXT_Folder) / "TestingList/index_endEstolic_test_id_frames24_25_phi.txt" #Path(TXT_Folder) / "TestingList/  .txt"





        train_files, test_id_files, test_ood_files, test_id_lv_seg_files, test_ood_lv_seg_files, \
        train_index_endEstolic_files, test_id_index_endEstolic_files, test_ood_index_endEstolic_files \
        = build_splits_from_txt(
            split,
            train_txt = train_txt_path,
            test_id_txt = test_id_txt_path,
            test_ood_txt = test_ood_txt_path,
            test_id_lv_seg_txt = test_id_lv_seg_txt_path,
            test_ood_lv_seg_txt = test_ood_lv_seg_txt_path,
            train_index_endEstolic_text = train_index_endEstolic_path,
            test_id_index_endEstolic_text = test_id_index_endEstolic_path,
            test_ood_index_endEstolic_text = test_ood_index_endEstolic_path
        )

        # self.files = files
        # self.lv_seg_files = lv_seg_files
        if split == "train":
            self.files = train_files
            self.lv_seg_files = None
            self.index_endEstolic_files = train_index_endEstolic_files

        elif split == "valid":
            self.files = test_id_files
            self.lv_seg_files = test_id_lv_seg_files
            self.index_endEstolic_files = test_id_index_endEstolic_files

        elif "test_ood" in split:
            self.files = test_ood_files
            self.lv_seg_files = test_ood_lv_seg_files
            self.index_endEstolic_files = test_ood_index_endEstolic_files
            # print(len(self.files), len(self.lv_seg_files),"~~~~~~~~~~~~~~~~ \n\n")
            # assert 4>555
        
        elif split == "test_id":
            self.files = test_id_files
            self.lv_seg_files = test_id_lv_seg_files
            self.index_endEstolic_files = test_id_index_endEstolic_files
            
        if FileLen is not None:
            self.files = self.files[Start_Subject_ID:Start_Subject_ID+FileLen]
            self.index_endEstolic_files = self.index_endEstolic_files[Start_Subject_ID:Start_Subject_ID+FileLen]
            if self.lv_seg_files is not None:
                self.lv_seg_files = self.lv_seg_files[Start_Subject_ID:Start_Subject_ID+FileLen]
    
        self.normalize = normalize
        self.resize = resize
        self.dtype = dtype
        self.drop_first_frame = drop_first_frame
        # self.Start_Subject_ID = Start_Subject_ID
       


        """ if "test" in split:
            ### ID  ###
            # selected_index = [32,122,155,193,227,255,299]   #7  dice = 0.4-0.5
            # selected_index = [4,6,34,35,44,73,77]  # dice = 0.5-0.6
            # selected_index = [29,84,89,96,143,156]  #6  dice = 0-0.4
            selected_index = [22,39,82,112,115]  # dice = 0.6-0.7
            selected_index = [13,79,111,83,104]  # dice = 0.9-1.0

            selected_index = [22,39,82,112,79,111,104] # dice = 0.6-0.7 +  dice = 0.9-1.0
            selected_index = [22,39,82,112,79,111,104] # dice = 0.6-0.7 +  dice = 0.9-1.0
            #selected_index = [2,5,7,16,17,26, 29]

            selected_index = [2,4,5,7,8,9,15]  #Group1 where TwoStages worse than TLRN
            selected_index = [16,35,36,89,113,115,299]  #Group2 sever outliers 
            selected_index = [10,11,12,19,24,39] #Group3 where TwoStages better than TLRN at most frames
            
            
            ##  Normal Groups  ##
            selected_index = [0,1,2,3,4,5,7]
            selected_index = [18,33,39,47,49,53,66,71]
            # selected_index = [92,111,142,144,147,156]

            ### OOD-ACDC  ###
            # selected_index = [202,635,691,697,927,957]   #7  dice = 0-0.5
            # selected_index = [379,453,583,638,678,701,714,735,807,835,863,891,912,956,1050,1061,1064,1065] #dice = 0.5-0.6
            # selected_index = [0,1,2,3,4,5,6] #dice = 0.6-0.7
            # selected_index = [12,13,21,23]   ##dice = 0.7-0.8
            # selected_index = [28,63,81,162,177,178]  ##dice = 0.9-1.0
            files_seleced = []
            index_endEstolic_files_selected = []
            lv_seg_files_selected = []

            for idx in selected_index:
                files_seleced.append(self.files[idx])
                index_endEstolic_files_selected.append(self.index_endEstolic_files[idx])
                if self.lv_seg_files is not None:
                    lv_seg_files_selected.append(self.lv_seg_files[idx])
            self.files = files_seleced
            self.index_endEstolic_files = index_endEstolic_files_selected
            if self.lv_seg_files is not None:
                self.lv_seg_files = lv_seg_files_selected
 """




    def __len__(self):
        return len(self.files)

    def _normalize(self, vol: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return vol
        vol = vol.astype(np.float32)
        if self.normalize == "zscore":
            # m = np.mean(vol)
            # s = np.std(vol)
            # if s < 1e-6:
            #     return vol - m
            # return (vol - m) / s
            vol = self.minmax(vol)
            return self.zscore_per_slice(vol)
        
        elif self.normalize == "percentile":
            vol = self.minmax(vol)
            return self.percentile_5_95_per_slice(vol)
        
        elif self.normalize == "histmatch":
            vol = self.minmax(vol)
            return self.global_histmatch(vol)

        elif self.normalize == "minmax":
            vmin = np.min(vol)
            vmax = np.max(vol)
            if abs(vmax - vmin) < 1e-6:
                return vol - vmin
            return (vol - vmin) / (vmax - vmin)
        else:
            return vol
    
    def minmax(self, vol: np.ndarray) -> np.ndarray:
        vmin = np.min(vol)
        vmax = np.max(vol)
        if abs(vmax - vmin) < 1e-6:
            return vol - vmin
        return (vol - vmin) / (vmax - vmin)


    def zscore_per_slice(self, volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Per-slice z-score normalization.

        Parameters
        ----------
        volume : np.ndarray
            Input array of shape (T, H, W).
        eps : float
            Small constant to avoid division by zero.

        Returns
        -------
        np.ndarray
            Z-score normalized volume with the same shape (T, H, W),
            where each slice volume[t] is normalized using its own mean/std.
        """
        vol = volume.astype(np.float32, copy=False)
        T, H, W = vol.shape
        out = np.zeros_like(vol, dtype=np.float32)

        for t in range(T):
            sl = vol[t]
            m = sl.mean()
            s = sl.std()
            out[t] = (sl - m) / (s + eps)

        return out
    import numpy as np



    def percentile_5_95_per_slice(self, volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Vectorized per-slice percentile (5–95) normalization.
        volume: (T, H, W)
        """
        vol = volume.astype(np.float32, copy=False)  # (T, H, W)

        # Compute percentiles over spatial dims for all T slices at once
        # p has shape (2, T)
        p = np.percentile(vol, [5, 95], axis=(1, 2))
        p5 = p[0][:, None, None]   # (T, 1, 1)
        p95 = p[1][:, None, None]  # (T, 1, 1)

        # Broadcast over H, W
        vol_clip = np.clip(vol, p5, p95)
        out = (vol_clip - p5) / (p95 - p5 + eps)

        # assert 4>111
        return out.astype(np.float32)



    def percentile_5_95_per_slice_expensive(self, volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Per-slice percentile (5–95) normalization.

        Parameters
        ----------
        volume : np.ndarray
            Input array of shape (T, H, W).
        eps : float
            Small epsilon to avoid division by zero.

        Returns
        -------
        np.ndarray
            Percentile-normalized volume with the same shape (T, H, W).
            Each slice is normalized separately.
        """
        vol = volume.astype(np.float32, copy=False)
        T, H, W = vol.shape
        out = np.zeros_like(vol, dtype=np.float32)
        for t in range(T):
            sl = vol[t]

            p5 = np.percentile(sl, 5)
            p95 = np.percentile(sl, 95)

            sl_clip = np.clip(sl, p5, p95)
            sl_norm = (sl_clip - p5) / (p95 - p5 + eps)

            out[t] = sl_norm
        return out


    def global_histmatch(self, volume: np.ndarray) -> np.ndarray:
        """
        Global histogram matching: each slice is matched to the mean slice
        (global reference) computed over time.

        Parameters
        ----------
        volume : np.ndarray
            Input array of shape (T, H, W).

        Returns
        -------
        np.ndarray
            Volume of shape (T, H, W) after global histogram matching,
            where each slice is matched to the global reference slice.
        """
        vol = volume.astype(np.float32, copy=False)
        # Build a 2D global reference slice by averaging all frames
        global_ref = vol.mean(axis=0)  # shape: (H, W)

        T, H, W = vol.shape
        out = np.zeros_like(vol, dtype=np.float32)

        for t in range(T):
            tar = vol[t]
            matched = match_histograms(tar, global_ref, channel_axis=None)
            out[t] = matched

        return out


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # idx =  + idx
        # assert 4>444
        path = self.files[idx]
        index_endEstolic = self.index_endEstolic_files[idx]
        # print("idx:   ", idx, index_endEstolic)
        

        lv_seg_path = self.lv_seg_files[idx] if self.lv_seg_files else None

        # 1) Load with SimpleITK → numpy (D, H, W). We treat D as T (frames).
        if not os.path.isfile(path) and (self.split == 'train' or self.split == 'valid'):          # 或者用 os.path.exists(path)
            path = path.replace("DataPd5", "Data")
            if lv_seg_path:
                lv_seg_path = lv_seg_path.replace("DataPd5", "Data")
            

        img = sitk.ReadImage(path)
        vol = sitk.GetArrayFromImage(img)  # shape: (T, H, W)
        

        index_endEstolic2 = None
        if lv_seg_path:
            if lv_seg_path.endswith(".gz") or lv_seg_path.endswith(".nii"):
                lv_seg_img = sitk.ReadImage(lv_seg_path)
                lv_seg = sitk.GetArrayFromImage(lv_seg_img)  # shape: (T, H, W)
            elif lv_seg_path.endswith(".mat"):
                lv_seg, index_endEstolic2 = load_mat_and_expand(lv_seg_path, target_Frames = vol.shape[0])  # shape: (T, H, W)
        else:
            lv_seg = None

        # 1) Load with SimpleITK → numpy (D, H, W). We treat D as T (frames).
        if vol.ndim != 3:
            raise ValueError(f"Expected a 3D volume, got shape {vol.shape} for {path}")
        vol = np.nan_to_num(vol, copy=False).astype(np.float32, copy=False)

        # 2) Normalize
        vol = self._normalize(vol)  # still (T, H, W), float32








        vol = vol[self.drop_first_frame:]
        if lv_seg is not None:
            lv_seg = lv_seg[self.drop_first_frame:]
        index_endEstolic -= self.drop_first_frame
        # print("idx:   ", idx, index_endEstolic)
        # 3) Pick a random 15-frame segment (pad with edge frames if T < 15)
        T, H, W = vol.shape
        # target_T = 15



        target_T = self.target_T

        # Temporal resampling to map the full cardiac cycle to target_T frames
        # e.g., if T=24 and target_T=15: frame 0->0, frame 23->14, frame 12->~7
        if T != target_T:
            # Resample volume using linear interpolation along temporal axis
            # Reshape: (T, H, W) -> (1, H*W, T) for 1D temporal interpolation
            vol_tensor = torch.from_numpy(vol).float()  # (T, H, W)
            vol_tensor = vol_tensor.permute(1, 2, 0).reshape(1, H * W, T)  # (1, H*W, T)
            vol_tensor = F.interpolate(vol_tensor, size=target_T, mode='linear', align_corners=True)
            vol = vol_tensor.reshape(H, W, target_T).permute(2, 0, 1).numpy()  # (target_T, H, W)

            # Remap index_endEstolic proportionally to the new temporal scale
            if T > 1:
                index_endEstolic = round(index_endEstolic * (target_T - 1) / (T - 1))

            # For segmentation, use nearest neighbor to preserve discrete label values
            if lv_seg is not None:
                lv_seg_tensor = torch.from_numpy(lv_seg).float()
                lv_seg_tensor = lv_seg_tensor.permute(1, 2, 0).reshape(1, H * W, T)
                lv_seg_tensor = F.interpolate(lv_seg_tensor, size=target_T, mode='nearest')
                lv_seg = lv_seg_tensor.reshape(H, W, target_T).permute(2, 0, 1).numpy()

        # 4) Reorder to (T, W, H) as requested
        vol = np.transpose(vol, (0, 2, 1))  # (15, W, H)
        if lv_seg is not None:
            lv_seg = np.transpose(lv_seg, (0, 2, 1))

        # 5) Convert to a torch tensor
        vol_th = torch.from_numpy(np.ascontiguousarray(vol)).to(self.dtype)  # (15, W, H)
        if lv_seg is not None:  
            lv_seg_th = torch.from_numpy(np.ascontiguousarray(lv_seg)).to(torch.float32)  # (15, W, H)

        # 6) Resize each frame to (W' = resize[0], H' = resize[1]) if requested
        #    F.interpolate expects (N, C, H, W); treat T as the batch dimension and add channel=1.
        if self.resize is not None:
            out_w = int(self.resize[0])
            out_h = int(self.resize[1])
            # Current: (T, W, H). Convert to (T, 1, H, W) → resize → back to (T, W, H)
            vol_th = vol_th.permute(0, 2, 1)            # (T, H, W)
            vol_th = vol_th.unsqueeze(1)                # (T, 1, H, W)
            vol_th = F.interpolate(vol_th, size=(out_h, out_w), mode="bilinear", align_corners=False)
            vol_th = vol_th.squeeze(1).permute(0, 2, 1) # (T, W, H)

            if lv_seg is not None:
                lv_seg_th = lv_seg_th.permute(0, 2, 1)
                lv_seg_th = lv_seg_th.unsqueeze(1)
                lv_seg_th = F.interpolate(lv_seg_th, size=(out_h, out_w), mode="nearest")
                lv_seg_th = lv_seg_th.squeeze(1).permute(0, 2, 1) # (T, W, H)

        vol_endEstolic = vol[index_endEstolic:index_endEstolic+1]
        if lv_seg is not None:
            lv_seg_endEstolic = lv_seg_th[index_endEstolic:index_endEstolic+1]

        # print("idx:   ", idx, index_endEstolic)
        # print("\n\n")
        if index_endEstolic2 is None:
            index_endEstolic2 = index_endEstolic

        return {
            "series": vol_th,               # (15, resize[0], resize[1]) if resized; otherwise (15, W, H)
            "lv_seg": lv_seg_th if lv_seg is not None else None,  # (15, resize[0], resize[1]) if resized; otherwise (15, W, H)
            "index_endEstolic": torch.tensor([index_endEstolic]), 
            "index_endEstolic2": torch.tensor([index_endEstolic2]), 
            "idx": torch.tensor([idx]),
            "vol_endEstolic": vol_endEstolic,
            "lv_seg_endEstolic": lv_seg_endEstolic if lv_seg is not None else None,
            "path": path,
            "orig_shape": (T, W, H),        # Original (before segment), reported as (T, W, H)
        }









def _read_txt_paths(
    txt_path: Path,
    *,
    root: Optional[Path] = None,
    verify_paths: bool = False,
    verbose: bool = True,
) -> List[str]:
    """
    从 txt 读取路径列表。
    - 忽略空行与以 '#' 开头的注释
    - 去重但保序
    - root: 若提供，则把相对路径拼到 root 下
    - verify_paths: 若 True，检查路径是否存在，不存在则打印告警并跳过
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"List file not found: {txt_path}")

    seen = set()
    out: List[str] = []
    missing: List[str] = []

    with txt_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    length = len(lines)
    print("length   ", length, txt_path)


    with txt_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            if root is not None and not p.is_absolute():
                p = root / p
            p = p.expanduser()

            # 去重（保序）
            key = str(p)

            """ if key in seen:
                continue
            seen.add(key)
            if verify_paths and not p.exists():
                missing.append(key)
                continue """
            
            
            out.append(key)

    if verify_paths and missing and verbose:
        print(f"[WARN] {len(missing)} missing files listed in {txt_path}:")
        for m in missing[:10]:
            print("   -", m)
        if len(missing) > 10:
            print(f"   ... and {len(missing)-10} more")
    return out



def _read_index_endEstolic(
    txt_path: Path,
) -> List[str]:
    """
    从 txt 读取路径列表。
    - 忽略空行与以 '#' 开头的注释
    - 去重但保序
    - root: 若提供，则把相对路径拼到 root 下
    - verify_paths: 若 True，检查路径是否存在，不存在则打印告警并跳过
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"List file not found: {txt_path}")
    out: List[str] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            out.append(int(line))
    return out



def load_splits_from_txt(
    split,
    train_txt: Path,
    test_id_txt: Path,
    test_ood_txt: Path,
    test_id_lv_seg_txt: Path,
    test_ood_lv_seg_txt: Path,
    train_index_endEstolic_text: Path,
    test_id_index_endEstolic_text: Path,
    test_ood_index_endEstolic_text: Path,

    *,
    root: Optional[Path] = None,
    verify_paths: bool = False,
    verbose: bool = True,
):
    """
    读取五个 split 列表并打包返回。
    - root: 为相对路径添加统一前缀（可选）
    - verify_paths: 校验文件是否存在（可选）
    """

    train_files = _read_txt_paths(train_txt, root=root, verify_paths=verify_paths, verbose=verbose)
    test_id_files = _read_txt_paths(test_id_txt, root=root, verify_paths=verify_paths, verbose=verbose)
    test_ood_files = _read_txt_paths(test_ood_txt, root=root, verify_paths=verify_paths, verbose=verbose)
    test_id_lv_seg_files = _read_txt_paths(test_id_lv_seg_txt, root=root, verify_paths=verify_paths, verbose=verbose)
    test_ood_lv_seg_files = _read_txt_paths(test_ood_lv_seg_txt, root=root, verify_paths=verify_paths, verbose=verbose)

    train_index_endEstolic_files = _read_index_endEstolic(train_index_endEstolic_text)
    test_id_index_endEstolic_files = _read_index_endEstolic(test_id_index_endEstolic_text)
    test_ood_index_endEstolic_files = _read_index_endEstolic(test_ood_index_endEstolic_text)
    

    
    print(f"\n\n[SPLIT] {split}\n",  test_ood_txt, "\n", test_ood_lv_seg_txt, "\n", test_ood_index_endEstolic_text, "\n", 
          f"| Train:    {len(train_files)} | Train-LV: None | Train-ES:    {len(train_index_endEstolic_files)}", "\n",
          f"| Test-ID:  {len(test_id_files)} | Test-ID-LV:  {len(test_id_lv_seg_files)}  | Test-ID-ES:  {len(test_id_index_endEstolic_files)}", "\n",
          f"| Test-OOD: {len(test_ood_files)} | Test-OOD-LV: {len(test_ood_lv_seg_files)} | Test-OOD-ES: {len(test_ood_index_endEstolic_files)}", "\n\n")
   

    return {
        "train_files": train_files,
        "test_id_files": test_id_files,
        "test_ood_files": test_ood_files,
        "test_id_lv_seg_files": test_id_lv_seg_files,
        "test_ood_lv_seg_files": test_ood_lv_seg_files,
        "train_index_endEstolic_files": train_index_endEstolic_files,
        "test_id_index_endEstolic_files": test_id_index_endEstolic_files,
        "test_ood_index_endEstolic_files": test_ood_index_endEstolic_files
    }