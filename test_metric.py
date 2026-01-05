import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
import torch
import lpips
import pyiqa
import numpy as np
from einops import rearrange
from utils import util_image

def load_image(path):
    img = util_image.imread(str(path), chn='rgb', dtype='float32')
    img = torch.from_numpy(img)
    img = rearrange(img, 'h w c -> 1 c h w')
    return img

def to_01(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

lpips_m = lpips.LPIPS(net="vgg").eval()
clip_m = pyiqa.create_metric("clipiqa")
musiq_m = pyiqa.create_metric("musiq")

results = {}

for tag in ["teacher", "student"]:
    scores = {"lpips": [], "clip": [], "musiq": []}
    img_dir = Path(f"sr_outputs3/{tag}")

    for p in img_dir.iterdir():
        img = load_image(p)
        img01 = to_01(img)

        scores["lpips"].append(lpips_m(img, img).item())  # dummy self-compare
        scores["clip"].append(clip_m(img01).item())
        scores["musiq"].append(musiq_m(img01).item())

    results[tag] = {k: float(np.mean(v)) for k, v in scores.items()}

# 출력
print("\n===== METRIC RESULT =====")
for k, v in results.items():
    print(
        f"{k.upper()} | "
        f"LPIPS {v['lpips']:.4f} | "
        f"CLIP {v['clip']:.4f} | "
        f"MUSIQ {v['musiq']:.4f}"
    )

# 파일 저장
with open("metric_result.txt", "w") as f:
    for k, v in results.items():
        f.write(
            f"{k.upper()} | "
            f"LPIPS {v['lpips']:.4f} | "
            f"CLIP {v['clip']:.4f} | "
            f"MUSIQ {v['musiq']:.4f}\n"
        )
