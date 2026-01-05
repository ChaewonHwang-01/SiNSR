import os, sys, math, time, random, datetime, functools
import lpips
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange
import copy
from datapipe.datasets import create_dataset

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import util_net, util_common, util_image
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt


# =====================================================
# ✅ 멀티프로세싱 안전 설정
# =====================================================
try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


# =====================================================
# ✅ TrainerBase (기반 클래스)
# =====================================================
class TrainerBase:
    def __init__(self, configs):
        self.configs = configs
        self.setup_dist()
        self.setup_seed()

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("⚠️ GPU 없음 — CPU 모드로 실행합니다.")
            self.num_gpus = 1
            self.rank = 0
            return

        # 분산 설정
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        if num_gpus > 1:
            rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                timeout=datetime.timedelta(seconds=3600),
                backend="nccl",
                init_method="env://",
            )
            self.rank = rank
        else:
            self.rank = 0

        self.num_gpus = num_gpus

    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get("seed", 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.get("global_seeding", True)

        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
        else:
            save_dir = Path(self.configs.save_dir) / datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        if self.rank == 0:
            logtxet_path = save_dir / "training.log"
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode="a")
            self.logger.add(sys.stdout, format="{message}", level="INFO")

            if self.configs.train.save_images:
                image_dir = save_dir / "images"
                (image_dir / "train").mkdir(parents=True, exist_ok=True)
                (image_dir / "val").mkdir(parents=True, exist_ok=True)
                self.image_dir = image_dir

            ckpt_dir = save_dir / "ckpts"
            ckpt_dir.mkdir(exist_ok=True)
            self.ckpt_dir = ckpt_dir

            if hasattr(self, "ema_rate"):
                ema_ckpt_dir = save_dir / "ema_ckpts"
                ema_ckpt_dir.mkdir(exist_ok=True)
                self.ema_ckpt_dir = ema_ckpt_dir

            self.logger.info(OmegaConf.to_yaml(self.configs))

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True:
                yield from loader

        datasets = {"train": create_dataset(self.configs.data.get("train", dict))}
        if hasattr(self.configs.data, "val") and self.rank == 0:
            datasets["val"] = create_dataset(self.configs.data.get("val", dict))
        if self.rank == 0:
            for phase in datasets.keys():
                self.logger.info(f"Dataset [{phase}] size: {len(datasets[phase])}")

        # Sampler
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                datasets["train"],
                num_replicas=self.num_gpus,
                rank=self.rank,
            )
        else:
            sampler = None

        num_workers = self.configs.train.get("num_workers", 4)
        prefetch_factor = None  # ✅ 멀티프로세싱 문제 방지

        dataloaders = {
            "train": _wrap_loader(
                udata.DataLoader(
                    datasets["train"],
                    batch_size=self.configs.train.batch[0] // max(1, self.num_gpus),
                    shuffle=False if self.num_gpus > 1 else True,
                    drop_last=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    prefetch_factor=prefetch_factor,
                    worker_init_fn=my_worker_init_fn,
                    sampler=sampler,
                )
            )
        }

        if hasattr(self.configs.data, "val") and self.rank == 0:
            dataloaders["val"] = udata.DataLoader(
                datasets["val"],
                batch_size=self.configs.train.batch[1],
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler


# =====================================================
# ✅ 이하 TrainerDifIR, TrainerDistillDifIR 등 기존 내용 그대로 유지
# =====================================================
# (중략 — 위에서 제공된 너의 원본 내용과 동일하게 두면 됨)
# 아래 함수들은 그대로 복사하면 됨.
# - setup_optimizaton
# - build_model
# - training_step
# - validation
# - update_ema_model
# - log_step_train
# - TrainerDifIR / TrainerDistillDifIR 정의
# - replace_nan_in_batch / my_worker_init_fn
# =====================================================

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
