import os
from pathlib import Path
import torch
from omegaconf import OmegaConf
from trainer1 import TrainerDistillDifIR  # ✅ 올바른 클래스
import torch.distributed as dist  # Trainer 내부에서만 사용

def main():
    config_path = "configs/SinSR.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"⚠️ 설정 파일이 존재하지 않습니다: {config_path}")

    cfg = OmegaConf.load(config_path)
    print("✅ Loaded config from:", config_path)

    # ---------------------------------------------------
    #  🔥 Trainer 내부에서 model 생성 + device 할당 + DDP wrap 책임지도록 둠
    #  main에서는 model을 직접 다루지 않는다
    # ---------------------------------------------------
    trainer = TrainerDistillDifIR(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
