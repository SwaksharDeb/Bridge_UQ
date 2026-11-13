"""
Load the pretrained TLRN registration backbone and the CINE data loaders.

Everything is resolved relative to this repository:

    configs/TLRN/basic.yaml + configs/TLRN/<experiment>.yaml   model / data config
    checkpoints/TLRN/<experiment>/*.ckpt                       pretrained weights
    dataload/TrainingList, dataload/TestingList                subject lists
    data/                                                      image data root

Overrides (environment variables):
    BRIDGEUQ_CKPT        path to the registration checkpoint to load
    BRIDGEUQ_DATA_ROOT   image data root (default: <repo>/data)
    BRIDGEUQ_LIST_DIR    folder holding TrainingList/ and TestingList/
                         (default: <repo>/dataload)
"""

import glob
import os

from box import Box
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from utils.utils import instantiate_from_config, merge_yaml_files
from utils.utils_train import load_checkpoints

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs")


def _find_checkpoint(ckpt_dir):
    """Best val_mse checkpoint, else best val_loss, else the latest epoch."""
    for pattern in ("epoch=*-val_mse=*.ckpt", "epoch=*-val_loss=*.ckpt", "epoch=*.ckpt", "*.ckpt"):
        found = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
        if found:
            return found[-1]
    return None


def load_model_and_data(combine_acdc=True):
    """
    Load the TLRN model and train/test data loaders.

    Args:
        combine_acdc: If True (default), use the model trained on the original
            1200-sample training set plus the 862-sample ACDC training set
            (2062 total). The combined file lists must exist at:
              TrainingList/train_1200_plus_ACDC.txt
              TrainingList/index_endEstolic_train_1200_plus_ACDC.txt

    Returns:
        model: Loaded TLRN model
        train_loader: Training data loader
        test_loader: Test data loader
        config: Configuration object
    """
    if combine_acdc:
        setobj = Box({
            "name": "CINE",
            "model": "TLRN",
            "editfile": "basic_MSE_Penp_img1200Reg0.03_plus_ACDC"
        })
        print("Using COMBINED dataset: original 1200 + ACDC 862 = 2062 samples")
    else:
        setobj = Box({
            "name": "CINE",
            "model": "TLRN",
            "editfile": "basic_MSE_Penp_img1200Reg0.03"
        })
        print("Using ORIGINAL dataset: 1200 samples only")

    workdir = os.path.join(OUTPUT_ROOT, setobj.model, setobj.editfile)
    os.makedirs(workdir, exist_ok=True)

    config_dir = os.path.join(REPO_ROOT, "configs", setobj.model)
    config_file_basic = os.path.join(config_dir, "basic.yaml")
    config_file_edit = os.path.join(config_dir, f"{setobj.editfile}.yaml")
    config_file = os.path.join(workdir, "config_merged.yaml")
    merge_yaml_files(config_file_basic, config_file_edit, config_file)

    seed_everything(20250101)
    config = OmegaConf.load(config_file)

    ckpt_dir = os.path.join(REPO_ROOT, "checkpoints", setobj.model, setobj.editfile)
    ckpt = os.environ.get("BRIDGEUQ_CKPT") or _find_checkpoint(ckpt_dir)
    if ckpt is None or not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"No pretrained {setobj.model} checkpoint found.\n"
            f"  looked in : {ckpt_dir}\n"
            f"  BRIDGEUQ_CKPT = {os.environ.get('BRIDGEUQ_CKPT')!r}\n"
            f"Put the registration checkpoint (*.ckpt) in that folder, or point "
            f"BRIDGEUQ_CKPT at it.")
    config.model.pretrained_checkpoint = ckpt
    print(f"Registration checkpoint: {ckpt}")

    config.model.params.workdir = workdir
    config.model.params.setobj_info = f"{setobj.name}_{setobj.model}_{setobj.editfile}"
    config.model.params.test_split = config.data.params.test.params.split
    config.model.params.Start_Subject_ID = 0

    print("Loading model...")
    model = instantiate_from_config(config.model)
    load_checkpoints(model, config.model)
    model.eval()
    model.cuda()

    print("Loading data...")
    data = instantiate_from_config(config.data)
    data.setup()
    train_loader = data.train_dataloader()
    test_loader = data.test_dataloader()

    print(f"  Train loader: {len(train_loader)} batches")
    print(f"  Test loader: {len(test_loader)} batches")

    return model, train_loader, test_loader, config
