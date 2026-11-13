import os, re
from omegaconf import OmegaConf
import logging
mainlogger = logging.getLogger('mainlogger')
import torch
from collections import OrderedDict

def init_workspace(name, logdir, model_config, lightning_config, rank=0):
    workdir = os.path.join(logdir, name)
    ckptdir = os.path.join(workdir, "checkpoints")
    cfgdir = os.path.join(workdir, "configs")
    loginfo = os.path.join(workdir, "loginfo")

    print("workdir:",  workdir, "\n")
    print("ckptdir:",  ckptdir, "\n")
    print("cfgdir:",  cfgdir, "\n")
    print("loginfo:",  loginfo, "\n")

    # Create logdirs and save configs (all ranks will do to avoid missing directory error if rank:0 is slower)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(loginfo, exist_ok=True)

    if rank == 0:
        if "callbacks" in lightning_config and 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
            os.makedirs(os.path.join(ckptdir, 'trainstep_checkpoints'), exist_ok=True)
        OmegaConf.save(model_config, os.path.join(cfgdir, "model.yaml"))
        OmegaConf.save(OmegaConf.create({"lightning": lightning_config}), os.path.join(cfgdir, "lightning.yaml"))
    return workdir, ckptdir, cfgdir, loginfo

def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None

def get_trainer_callbacks(lightning_config, config, logdir, ckptdir, logger, stage_name=""):
    # #print("\n\n\n")
    # #print(lightning_config.callbacks)
    # assert 3>222
    default_callbacks_cfg = {}
    if "model_checkpoint" in lightning_config.callbacks:
        default_callbacks_cfg["model_checkpoint"] = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch}",
                "verbose": True,
                "save_last": False,
            }
        }
    if "metrics_val_loss_checkpoint" in lightning_config.callbacks:
        default_callbacks_cfg["metrics_val_loss_checkpoint"] = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
            }
        }

    if "metrics_mse_loss_checkpoint" in lightning_config.callbacks:
        default_callbacks_cfg["metrics_mse_loss_checkpoint"] = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
            }
        }



    """ default_callbacks_cfg = {
        "model_checkpoint": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch}",
                "verbose": True,
                "save_last": False,
            }
        },
        "metrics_val_loss_checkpoint":{
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
            }
        },
        "metrics_mse_loss_checkpoint":{
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
            }
        },

        # "batch_logger": {
        #     "target": "callbacks.ImageLogger",
        #     "params": {
        #         "save_dir": logdir,
        #         "batch_frequency": 1000,
        #         "max_images": 4,
        #         "clamp": True,
        #     }
        # },    
        # "learning_rate_logger": {
        #     "target": "pytorch_lightning.callbacks.LearningRateMonitor",
        #     "params": {
        #         "logging_interval": "step",
        #         "log_momentum": False
        #     }
        # },
        "cuda_callback": {
            "target": "callbacks.CUDACallback"
        },
    }
 """
    ## optional setting for saving checkpoints
    monitor_metric = check_config_attribute(config.model.params, "monitor")
    if monitor_metric is not None:
        mainlogger.info(f"Monitoring {monitor_metric} as checkpoint metric.")
        default_callbacks_cfg["model_checkpoint"]["params"]["monitor"] = monitor_metric
        default_callbacks_cfg["model_checkpoint"]["params"]["save_top_k"] = 3
        default_callbacks_cfg["model_checkpoint"]["params"]["mode"] = "min"

    if 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
        mainlogger.info('Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint': {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                                                   'params': {
                                                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                                                        "filename": "{epoch}-{step}",
                                                        "verbose": True,
                                                        'save_top_k': -1,
                                                        'every_n_train_steps': 10000,
                                                        'save_weights_only': True
                                                    }
                                                }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    """ #print("\n\n")
    #print(callbacks_cfg)
    {
    "model_checkpoint": {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
        "dirpath": "/home/nellie/code/cvpr/Project/LightningTemplate/2026Experiments/CINE/outputs/TLRN/basic/checkpoints",
        "filename": "{epoch}-{step}",
        "verbose": true,
        "save_last": false,
        "every_n_epochs": 1,
        "save_weights_only": true
        }
    },
    "metrics_val_loss_checkpoint": {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
        "dirpath": "/home/nellie/code/cvpr/Project/LightningTemplate/2026Experiments/CINE/outputs/TLRN/basic/checkpoints",
        "monitor": "val_loss",
        "mode": "min",
        "filename": "{epoch}-{val_loss:.4f}",
        "save_top_k": 1,
        "save_weights_only": true
        }
    },
    "metrics_mse_loss_checkpoint": {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
        "dirpath": "/home/nellie/code/cvpr/Project/LightningTemplate/2026Experiments/CINE/outputs/TLRN/basic/checkpoints",
        "monitor": "val_mse",
        "mode": "min",
        "filename": "{epoch}-{val_mse:.4f}",
        "save_top_k": 1,
        "save_weights_only": true
        }
    }
    }
    assert 3>198 """

    return callbacks_cfg

# def get_trainer_logger(lightning_config, logdir, on_debug, stage_name=""):
#     default_logger_cfgs = {
#         "tensorboard": {
#             "target": "pytorch_lightning.loggers.TensorBoardLogger",
#             "params": {
#                 "save_dir": logdir,
#                 "name": "tensorboard",
#             }
#         },
#         "testtube": {
#             "target": "pytorch_lightning.loggers.CSVLogger",
#             "params": {
#                     "name": "testtube",
#                     "save_dir": logdir,
#                 }
#             },
#     }
#     os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)
#     default_logger_cfg = default_logger_cfgs["tensorboard"]
#     if "logger" in lightning_config:
#         logger_cfg = lightning_config.logger
#     else:
#         logger_cfg = OmegaConf.create()
#     logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
#     return logger_cfg
def get_trainer_logger(lightning_config, logdir, on_debug, stage_name=None):
    # 可选：给 stage 起个前缀
    if stage_name is not None:
        stage_suffix = f"-{stage_name}"       # 用在 logger name 上，比如 tensorboard-stage1
        # logdir_stage = os.path.join(logdir, stage_name)  # 用在 save_dir 上，比如 .../workdir/stage1
    else:
        stage_suffix = ""
        # logdir_stage = logdir

    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,              # 带 stage 子目录
                "name": f"tensorboard{stage_suffix}", # tensorboard-stage1 / tensorboard-stage2
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                "name": f"testtube{stage_suffix}",
                "save_dir": logdir,
            }
        },
    }

    # os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)

    default_logger_cfg = default_logger_cfgs["tensorboard"]

    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()

    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return logger_cfg







def get_trainer_strategy(lightning_config):
    default_strategy_dict = {
        "target": "pytorch_lightning.strategies.DDPShardedStrategy"    ##1x
        # "target": "pytorch_lightning.strategies.DDPStrategy"            ##2x
    }
    if "strategy" in lightning_config:
        strategy_cfg = lightning_config.strategy
        return strategy_cfg
    else:
        strategy_cfg = OmegaConf.create()

    strategy_cfg = OmegaConf.merge(default_strategy_dict, strategy_cfg)
    return strategy_cfg





# def load_unetdecoderhead_weights_from_unethead(
#     model, model_cfg, src_prefix="unet_head.", dst_prefix="unet2.", map_location="cpu"
# ):
def load_net1_weights(model, model_cfg, map_location="cpu"):
    """
    只加载 stage1 相关权重：
      - unet1.*   (先按老逻辑过滤出来)
      - residue_block1.*

    然后把 unet1 里「非 encoder」部分的权重拷贝到 UNetDecoderHead（假设挂在 model.unet2 下）：
      - unet1.decoder.*    -> unet2.decoder.*
      - unet1.remaining.*  -> unet2.remaining.*
      - unet1.flow.*       -> unet2.flow.*
      - unet1.upsampling.* -> unet2.upsampling.*
    并且明确 rule out encoder：unet1.encoder.* 不加载到 decoder head。

    注意：
    - 这里默认你的 decoder head 在 model 里叫 `unet2`（即 state_dict key 前缀是 "unet2."）
    - 过滤阶段仍然保持你原来的：只看 "unet1." 和 "residue_block1."
    """

    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint

        assert os.path.exists(pretrained_ckpt), \
            "Error: Stage1 checkpoint NOT found at: %s" % pretrained_ckpt
        mainlogger.info(">>> [Stage1->Decoder] Load partial weights from pretrained checkpoint")

        pl_sd = torch.load(pretrained_ckpt, map_location=map_location)

        # 先按照你原来的逻辑，把真正的 state_dict 抽出来
        try:
            if isinstance(pl_sd, dict) and 'state_dict' in pl_sd.keys():
                sd = pl_sd["state_dict"]
                mainlogger.info(">>> [Stage1->Decoder] Found 'state_dict' in checkpoint.")
            elif isinstance(pl_sd, dict) and 'registration' in pl_sd.keys():
                sd = pl_sd["registration"]
                mainlogger.info(">>> [Stage1->Decoder] Found 'registration' in checkpoint.")
            elif isinstance(pl_sd, dict) and 'module' in pl_sd.keys():
                # deepspeed 形式
                new_pl_sd = OrderedDict()
                for key in pl_sd['module'].keys():
                    new_pl_sd[key[16:]] = pl_sd['module'][key]
                sd = new_pl_sd
                mainlogger.info(">>> [Stage1->Decoder] Found 'module' in checkpoint (deepspeed style).")
            else:
                # 直接就是 state_dict
                sd = pl_sd
                mainlogger.info(">>> [Stage1->Decoder] Use checkpoint as plain state_dict.")
        except Exception as e:
            mainlogger.info(f">>> [Stage1->Decoder] Fallback: use checkpoint as plain state_dict due to error: {e}")
            sd = pl_sd

        model_sd = model.state_dict()

        # ============ 1) 保持你原来的 filter：先只挑 unet1 / residue_block1 ============
        filtered = OrderedDict()
        skipped_shape = []
        skipped_missing = []

        for k, v in sd.items():
            if k.startswith("unet1.") or k.startswith("residue_block1."):
                if k in model_sd:
                    if model_sd[k].shape == v.shape:
                        filtered[k] = v
                    else:
                        skipped_shape.append(k)
                else:
                    skipped_missing.append(k)

        # ============ 2) 把 unet1 的“非 encoder”权重映射到 unet2（decoder head） ============
        # 你说的 “copy the weight to unetdecoderhead，rule out encoder”
        mapped = OrderedDict()
        skipped_encoder = []
        skipped_map_missing = []
        skipped_map_shape = []

        for k, v in filtered.items():
            # 只对 unet1.* 做映射；residue_block1.* 按原样保留（如果你也需要）
            if k.startswith("unet1.encoder."):
                skipped_encoder.append(k)
                continue

            if k.startswith("unet1."):
                k2 = "unet2." + k[len("unet1."):]   # unet1.xxx -> unet2.xxx
            else:
                k2 = k  # residue_block1.* 之类

            if k2 in model_sd:
                if model_sd[k2].shape == v.shape:
                    mapped[k2] = v
                else:
                    skipped_map_shape.append((k, k2))
            else:
                skipped_map_missing.append((k, k2))

        # ============ 3) 更新并加载 ============
        model_sd.update(mapped)
        model.load_state_dict(model_sd, strict=False)

        mainlogger.info(
            f">>> [Stage1->Decoder] Loaded {len(mapped)} tensors into decoder head from: {pretrained_ckpt}"
        )
        if skipped_encoder:
            mainlogger.info(
                f">>> [Stage1->Decoder] Skipped {len(skipped_encoder)} encoder keys (ruled out)."
            )
        if skipped_shape:
            mainlogger.info(
                f">>> [Stage1->Decoder] (filter stage) Skipped {len(skipped_shape)} keys due to shape mismatch."
            )
        if skipped_missing:
            mainlogger.info(
                f">>> [Stage1->Decoder] (filter stage) Skipped {len(skipped_missing)} keys not found in current model."
            )
        if skipped_map_shape:
            mainlogger.info(
                f">>> [Stage1->Decoder] (map stage) Skipped {len(skipped_map_shape)} mapped keys due to shape mismatch."
            )
        if skipped_map_missing:
            mainlogger.info(
                f">>> [Stage1->Decoder] (map stage) Skipped {len(skipped_map_missing)} mapped keys not found in current model."
            )

    else:
        mainlogger.info(">>> [Stage1->Decoder] No pretrained_checkpoint found in config. Start from scratch.")

    return model






def load_net1_weights(model, model_cfg, map_location="cpu"):
    """
    只加载 stage1 相关权重：
      - unet1.*
      - residue_block1.*

    用法示例：
        model = load_net1_weights(model, config.model_stage1)
    其中 config.model_stage1 里要有 pretrained_checkpoint 字段。
    """


    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint

        assert os.path.exists(pretrained_ckpt), \
            "Error: Stage1 checkpoint NOT found at: %s" % pretrained_ckpt
        mainlogger.info(">>> [Stage1] Load partial weights from pretrained checkpoint")

        pl_sd = torch.load(pretrained_ckpt, map_location=map_location)
        # print(pl_sd['state_dict'].keys())
        # assert 3>129


        # 先按照你原来的逻辑，把真正的 state_dict 抽出来
        try:
            if isinstance(pl_sd, dict) and 'state_dict' in pl_sd.keys():
                sd = pl_sd["state_dict"]
                mainlogger.info(">>> [Stage1] Found 'state_dict' in checkpoint.")
            elif isinstance(pl_sd, dict) and 'registration' in pl_sd.keys():
                sd = pl_sd["registration"]
                mainlogger.info(">>> [Stage1] Found 'registration' in checkpoint.")
            elif isinstance(pl_sd, dict) and 'module' in pl_sd.keys():
                # deepspeed 形式
                new_pl_sd = OrderedDict()
                for key in pl_sd['module'].keys():
                    new_pl_sd[key[16:]] = pl_sd['module'][key]
                sd = new_pl_sd
                mainlogger.info(">>> [Stage1] Found 'module' in checkpoint (deepspeed style).")
            else:
                # 直接就是 state_dict
                sd = pl_sd
                mainlogger.info(">>> [Stage1] Use checkpoint as plain state_dict.")
        except Exception as e:
            mainlogger.info(f">>> [Stage1] Fallback: use checkpoint as plain state_dict due to error: {e}")
            sd = pl_sd

        # 只挑 stage1 相关的 key
        model_sd = model.state_dict()
        filtered = OrderedDict()
        skipped_shape = []
        skipped_missing = []

        for k, v in sd.items():
            if k.startswith("unet1.") or k.startswith("residue_block1."):
                if k in model_sd:
                    if model_sd[k].shape == v.shape:
                        filtered[k] = v
                    else:
                        skipped_shape.append(k)
                else:
                    skipped_missing.append(k)

        # 更新并加载
        model_sd.update(filtered)
        model.load_state_dict(model_sd, strict=False)

        mainlogger.info(
            f">>> [Stage1] Loaded {len(filtered)} tensors from: {pretrained_ckpt}"
        )
        if skipped_shape:
            mainlogger.info(
                f">>> [Stage1] Skipped {len(skipped_shape)} keys due to shape mismatch."
            )
        if skipped_missing:
            mainlogger.info(
                f">>> [Stage1] Skipped {len(skipped_missing)} keys not found in current model."
            )

    else:
        mainlogger.info(">>> [Stage1] No pretrained_checkpoint found in config. Start from scratch for stage1.")

    return model


def load_net2_weights(model, model_cfg, map_location="cpu"):
    """
    只加载 stage2 相关权重：
      - unet2.*
      - residue_block2.*
      - （可选）MSvf.*  如果你想一起加载，可以把下面注释打开

    用法示例：
        model = load_net2_weights(model, config.model_stage2)
    其中 config.model_stage2 里要有 pretrained_checkpoint 字段。
    """
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint

        assert os.path.exists(pretrained_ckpt), \
            "Error: Stage2 checkpoint NOT found at: %s" % pretrained_ckpt
        mainlogger.info(">>> [Stage2] Load partial weights from pretrained checkpoint")

        pl_sd = torch.load(pretrained_ckpt, map_location=map_location)

        # 抽 state_dict，逻辑跟上面保持一致
        try:
            if isinstance(pl_sd, dict) and 'state_dict' in pl_sd.keys():
                sd = pl_sd["state_dict"]
                mainlogger.info(">>> [Stage2] Found 'state_dict' in checkpoint.")
            elif isinstance(pl_sd, dict) and 'registration' in pl_sd.keys():
                sd = pl_sd["registration"]
                mainlogger.info(">>> [Stage2] Found 'registration' in checkpoint.")
            elif isinstance(pl_sd, dict) and 'module' in pl_sd.keys():
                new_pl_sd = OrderedDict()
                for key in pl_sd['module'].keys():
                    new_pl_sd[key[16:]] = pl_sd['module'][key]
                sd = new_pl_sd
                mainlogger.info(">>> [Stage2] Found 'module' in checkpoint (deepspeed style).")
            else:
                sd = pl_sd
                mainlogger.info(">>> [Stage2] Use checkpoint as plain state_dict.")
        except Exception as e:
            mainlogger.info(f">>> [Stage2] Fallback: use checkpoint as plain state_dict due to error: {e}")
            sd = pl_sd

        model_sd = model.state_dict()
        filtered = OrderedDict()
        skipped_shape = []
        skipped_missing = []

        for k, v in sd.items():
            if (
                k.startswith("unet2.")
                # or k.startswith("residue_block2.")
                # 或者你希望一起加载的话：
                # or k.startswith("MSvf.")
            ):
                if k in model_sd:
                    if model_sd[k].shape == v.shape:
                        filtered[k] = v
                    else:
                        skipped_shape.append(k)
                else:
                    skipped_missing.append(k)

        model_sd.update(filtered)
        model.load_state_dict(model_sd, strict=False)

        mainlogger.info(
            f">>> [Stage2] Loaded {len(filtered)} tensors from: {pretrained_ckpt}"
        )
        if skipped_shape:
            mainlogger.info(
                f">>> [Stage2] Skipped {len(skipped_shape)} keys due to shape mismatch."
            )
        if skipped_missing:
            mainlogger.info(
                f">>> [Stage2] Skipped {len(skipped_missing)} keys not found in current model."
            )

    else:
        mainlogger.info(">>> [Stage2] No pretrained_checkpoint found in config. Start from scratch for stage2.")

    return model











def load_checkpoints(model, model_cfg):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint        
        

        assert os.path.exists(pretrained_ckpt), "Error: Pre-trained checkpoint NOT found at:%s"%pretrained_ckpt
        mainlogger.info(">>> Load weights from pretrained checkpoint")

        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        # print(pl_sd.keys())
        # print("\n\n")
        # assert 4>398
       
        try:
            if 'state_dict' in pl_sd.keys():
                model.load_state_dict(pl_sd["state_dict"], strict=True)
                mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)
                print(f"\n\n\n  Successfully load the checkpoints: {  pretrained_ckpt  } \n\n\n")
            elif 'registration' in pl_sd.keys():
                model.load_state_dict(pl_sd["registration"], strict=True)
                mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)
                print(f"\n\n\n  Successfully load the checkpoints: {  pretrained_ckpt  } \n\n\n")
            else:
                # deepspeed
                new_pl_sd = OrderedDict()
                for key in pl_sd['module'].keys():
                    new_pl_sd[key[16:]]=pl_sd['module'][key]
                model.load_state_dict(new_pl_sd, strict=True)
                print(f"\n\n\n  Successfully load the checkpoints: {  pretrained_ckpt  } \n\n\n")
        except:
            model.load_state_dict(pl_sd)
    else:
        mainlogger.info(">>> Start training from scratch")
    return model


def load_checkpoints_unet(model, model_cfg):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), "Error: Pre-trained checkpoint NOT found at:%s"%pretrained_ckpt
        mainlogger.info(">>> Load weights from pretrained checkpoint")

        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        try:
            if 'state_dict' in pl_sd.keys():
                unet_weights = {k.replace("unet.", ""): v for k, v in pl_sd['state_dict'].items() if k.startswith("unet.")}
                model.unet.load_state_dict(unet_weights, strict=True)

                # model.load_state_dict(pl_sd["state_dict"], strict=True)
                mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)
            else:
                # deepspeed
                new_pl_sd = OrderedDict()
                for key in pl_sd['module'].keys():
                    new_pl_sd[key[16:]]=pl_sd['module'][key]
                model.load_state_dict(new_pl_sd, strict=True)
        except:
            model.load_state_dict(pl_sd)
    else:
        mainlogger.info(">>> Start training from scratch")
    return model


def load_checkpoints_without_residue(model, model_cfg):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), "Error: Pre-trained checkpoint NOT found at:%s"%pretrained_ckpt
        mainlogger.info(">>> Load weights from pretrained checkpoint")

        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        try:
            if 'state_dict' in pl_sd.keys():
                unet_weights = {k: v for k, v in pl_sd['state_dict'].items() if not k.startswith("residue_block.")}
                model.load_state_dict(unet_weights, strict=True)

                # model.load_state_dict(pl_sd["state_dict"], strict=True)
                mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)
            else:
                # deepspeed
                new_pl_sd = OrderedDict()
                for key in pl_sd['module'].keys():
                    new_pl_sd[key[16:]]=pl_sd['module'][key]
                model.load_state_dict(new_pl_sd, strict=True)
        except:
            model.load_state_dict(pl_sd)
    else:
        mainlogger.info(">>> Start training from scratch")
    return model


def load_checkpoints_nop(model, model_cfg):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), "Error: Pre-trained checkpoint NOT found at:%s"%pretrained_ckpt
        mainlogger.info(">>> Load weights from pretrained checkpoint")

        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")

        # #print("\n\n")
        # #print(pl_sd["state_dict"].keys())
        # #print("\n\n")
        # assert 3>198



        try:
            if 'state_dict' in pl_sd.keys():
                unet_weights = {k.replace("unet.", ""): v for k, v in pl_sd['state_dict'].items() if k.startswith("unet.")}
                nop_weights = {k.replace("nop.nop.", "nop."): v for k, v in pl_sd['state_dict'].items() if k.startswith("nop.")}
                # nop_weights = {k: v for k, v in pl_sd['state_dict'].items() if k.startswith("nop.")}
                
                #print("\n\n")
                #print(nop_weights)
                #print("\n\n")


                model.nop.load_state_dict(nop_weights, strict=True)

                # model.load_state_dict(pl_sd["state_dict"], strict=True)
                mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)
            else:
                # deepspeed
                new_pl_sd = OrderedDict()
                for key in pl_sd['module'].keys():
                    new_pl_sd[key[16:]]=pl_sd['module'][key]
                model.load_state_dict(new_pl_sd, strict=True)
        except:
            model.load_state_dict(pl_sd)
    else:
        mainlogger.info(">>> Start training from scratch")
    return model



def set_logger(logfile, name='mainlogger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s: %(message)s"))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger