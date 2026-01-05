# test_sr.py
import argparse
import inspect
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from einops import rearrange

from utils import util_image
from utils import util_common
from utils import util_net


# ------------------------------------------------------------
# I/O (normalize_input에 따라 처리)
# ------------------------------------------------------------
def load_image(path: Path, device: torch.device, normalize_input: bool) -> torch.Tensor:
    """
    util_image.imread -> [0,1] float32

    - normalize_input=True  : 입력을 [0,1] 그대로 유지 (diffusion 내부에서 normalize 처리하는 설계)
    - normalize_input=False : 기존처럼 [-1,1]로 변환
    """
    img = util_image.imread(str(path), chn="rgb", dtype="float32")  # [0,1]
    img = torch.from_numpy(img)
    img = rearrange(img, "h w c -> 1 c h w")
    if not normalize_input:
        img = (img - 0.5) / 0.5  # [-1,1]
    return img.to(device, non_blocking=True)


def save_image(tensor: torch.Tensor, path: Path, normalize_input: bool):
    """
    - normalize_input=False일 때만 [-1,1] -> [0,1] 복원
    - normalize_input=True이면 출력이 [0,1] domain인 설계가 많아 추가 변환 안 함
      (만약 네 diffusion 출력이 [-1,1]이라면 여기서 성능/색감이 이상해질 수 있으니 그땐 알려줘)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    img = tensor.detach().float().cpu().numpy()
    img = rearrange(img, "1 c h w -> h w c")

    if not normalize_input:
        img = img * 0.5 + 0.5

    img = img.clip(0, 1)
    util_image.imwrite(img, str(path))


# ------------------------------------------------------------
# Swin padding
# ------------------------------------------------------------
def pad_for_swin(x, window_size: int, downsample: int):
    _, _, h, w = x.shape
    base = window_size * downsample
    pad_h = (base - h % base) % base
    pad_w = (base - w % base) % base
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, pad_h, pad_w


def remove_padding(x, pad_h: int, pad_w: int):
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x


# ------------------------------------------------------------
# 핵심: Gradio demo와 동일하게 "스케줄 유지"
# - Teacher: diffusion 객체가 이미 steps=15로 생성되어 있어야 함.
#            => 여기서는 one_step=False만 정확히 전달
# - Student: one_step=True
# ------------------------------------------------------------
@torch.no_grad()
def run_inference(model, diffusion, autoencoder, img, *, is_teacher: bool):
    base_kwargs = dict(
        y=img,
        model=model,
        first_stage_model=autoencoder,
        noise=None,
        clip_denoised=True,
        model_kwargs={"lq": img},
        progress=False,
    )

    sig = inspect.signature(diffusion.ddim_sample_loop)
    params = sig.parameters

    if "one_step" not in params:
        raise RuntimeError(
            "diffusion.ddim_sample_loop에 one_step 인자가 없습니다. "
            "현재 diffusion 구현이 Gradio demo 방식(one_step True/False)을 지원하는지 확인 필요."
        )

    # ✅ teacher: multi-step (스케줄은 diffusion 생성 시 이미 15-step이어야 함)
    if is_teacher:
        return diffusion.ddim_sample_loop(**base_kwargs, one_step=False)

    # ✅ student: single-step
    return diffusion.ddim_sample_loop(**base_kwargs, one_step=True)


# ------------------------------------------------------------
# 타일 마스크/타일 SR (원래 코드 유지)
# ------------------------------------------------------------
def make_blend_mask(h: int, w: int, overlap: int, device: torch.device):
    if overlap <= 0:
        return torch.ones(1, 1, h, w, device=device)
    overlap = min(overlap, h // 2, w // 2)
    ramp = torch.linspace(0, 1, steps=overlap, device=device)

    yy = torch.ones(h, device=device)
    xx = torch.ones(w, device=device)
    yy[:overlap] = ramp
    yy[-overlap:] = torch.flip(ramp, dims=[0])
    xx[:overlap] = ramp
    xx[-overlap:] = torch.flip(ramp, dims=[0])

    mask2d = yy[:, None] * xx[None, :]
    return mask2d[None, None, :, :]


@torch.inference_mode()
def tiled_sr(
    img: torch.Tensor,       # (1,3,H,W)
    device: torch.device,
    model,
    autoencoder,
    diffusion,
    window_size: int,
    downsample: int,
    tile: int,
    overlap: int,
    use_amp: bool,
    scale: int,
    is_teacher: bool,
):
    _, _, H, W = img.shape
    stride = tile - overlap
    assert stride > 0, "tile must be > overlap"

    Hh, Wh = H * scale, W * scale
    out_acc = torch.zeros(1, 3, Hh, Wh, device=device, dtype=torch.float32)
    w_acc = torch.zeros(1, 1, Hh, Wh, device=device, dtype=torch.float32)

    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            y0_ = max(0, y1 - tile)
            x0_ = max(0, x1 - tile)

            patch = img[:, :, y0_:y1, x0_:x1]
            patch_pad, ph, pw = pad_for_swin(patch, window_size, downsample)

            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    sr = run_inference(model, diffusion, autoencoder, patch_pad, is_teacher=is_teacher)
            else:
                sr = run_inference(model, diffusion, autoencoder, patch_pad, is_teacher=is_teacher)

            sr = remove_padding(sr, ph * scale, pw * scale)

            hy0, hx0 = y0_ * scale, x0_ * scale
            hy1, hx1 = y1 * scale, x1 * scale

            th, tw = (y1 - y0_) * scale, (x1 - x0_) * scale
            if sr.shape[-2] != th or sr.shape[-1] != tw:
                sr = sr[:, :, :th, :tw]

            mask = make_blend_mask(
                h=sr.shape[-2],
                w=sr.shape[-1],
                overlap=overlap * scale,
                device=device,
            ).float()

            out_acc[:, :, hy0:hy1, hx0:hx1] += sr.float() * mask
            w_acc[:, :, hy0:hy1, hx0:hx1] += mask

            del patch, patch_pad, sr, mask

    out = out_acc / torch.clamp(w_acc, min=1e-6)
    return out


# ------------------------------------------------------------
# Config 로딩: Gradio get_configs() 스타일로 맞춤
# ------------------------------------------------------------
def load_cfg_and_weights(mode: str):
    """
    mode=teacher:
      - Gradio에서 ResShift 선택했을 때와 동일하게:
        configs/realsr_swinunet_realesrgan256.yaml 로드
        diffusion.steps=15 / sf=4 강제
        weight=weights/resshift_realsrx4_s15_v1.pth

    mode=student:
      - configs/SinSR.yaml 로드
      - weight=experiments/...ema_model_500000.pth
    """
    if mode == "teacher":
        cfg = OmegaConf.load("./configs/realsr_swinunet_realesrgan256.yaml")
        cfg.diffusion.params.steps = 15
        cfg.diffusion.params.sf = 4
        weight_path = "./weights/resshift_realsrx4_s15_v1.pth"
        out_sub = "teacher"
    else:
        cfg = OmegaConf.load("./configs/SinSR.yaml")
        weight_path = "./experiments/2025-12-10-12-55/ema_ckpts/ema_model_500000.pth"
        out_sub = "student"

    # scale은 cfg.sf를 최우선으로 사용 (Gradio demo 동일)
    scale = int(cfg.diffusion.params.sf)

    # normalize_input은 cfg에 맞춤 (성능 이슈 1순위)
    normalize_input = bool(cfg.diffusion.params.get("normalize_input", False))

    return cfg, weight_path, out_sub, scale, normalize_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["teacher", "student"], required=True)
    parser.add_argument("--gpu", type=int, required=True)

    # ✅ 폴더 기반 유지 (원래 흐름 유지)
    parser.add_argument("--test_dir", type=str, default="/home/kyujoonglee/SinSR_256/testdata/RealSet65")
    parser.add_argument("--out_root", type=str, default="sr_outputs3")

    # ✅ 타일 옵션은 없애지 않음 (기본은 타일 OFF)
    # parser.add_argument("--use_tile", action="store_true", help="Use tiled inference (default: False)")
    parser.add_argument("--tile", type=int, default=192)
    parser.add_argument("--overlap", type=int, default=32)

    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    assert args.gpu < torch.cuda.device_count(), f"GPU {args.gpu} not available"

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    is_teacher = (args.mode == "teacher")

    # ------------------------------------------------------------
    # cfg/weights (Gradio 스타일)
    # ------------------------------------------------------------
    cfg, weight_path, out_sub, scale, normalize_input = load_cfg_and_weights(args.mode)

    window_size = int(cfg.model.params.window_size)
    downsample = 8  # Swin 계열 UNet downsample 규칙(일반적으로 8)

    diffusion = util_common.get_obj_from_str(cfg.diffusion.target)(**cfg.diffusion.params)

    # 디버깅 출력 (스케줄 확인)
    print("[INFO] mode:", args.mode)
    print("[INFO] cfg.diffusion.params.steps:", cfg.diffusion.params.get("steps", None))
    if hasattr(diffusion, "use_timesteps"):
        try:
            uts = list(diffusion.use_timesteps)
            print("[INFO] len(use_timesteps):", len(uts), " head:", uts[:5], " tail:", uts[-5:])
        except Exception as e:
            print("[INFO] use_timesteps print failed:", e)
    print("[INFO] normalize_input:", normalize_input)
    print("[INFO] scale(sf):", scale)
    print("[INFO] use_tile: HARD TRUE")

    # ------------------------------------------------------------
    # autoencoder
    # ------------------------------------------------------------
    ae_cfg = cfg.autoencoder
    ae_state = torch.load(ae_cfg.ckpt_path, map_location="cpu")
    autoencoder = util_common.get_obj_from_str(ae_cfg.target)(**ae_cfg.params)
    autoencoder.load_state_dict(ae_state)
    autoencoder.eval().to(device)

    # ------------------------------------------------------------
    # model
    # ------------------------------------------------------------
    model = util_common.get_obj_from_str(cfg.model.target)(**cfg.model.params)
    state = torch.load(weight_path, map_location="cpu")
    util_net.reload_model(model, state)
    model.eval().to(device)

    # ------------------------------------------------------------
    # inputs
    # ------------------------------------------------------------
    test_dir = Path(args.test_dir)
    img_paths = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    out_dir = Path(args.out_root) / out_sub
    out_dir.mkdir(parents=True, exist_ok=True)

    use_amp = not args.no_amp

    for p in img_paths:
        img = load_image(p, device, normalize_input=normalize_input)

        # ✅ 무조건 타일 사용 (args.use_tile 분기 제거)
        sr = tiled_sr(
            img=img,
            device=device,
            model=model,
            autoencoder=autoencoder,
            diffusion=diffusion,
            window_size=window_size,
            downsample=downsample,
            tile=args.tile,
            overlap=args.overlap,
            use_amp=use_amp,
            scale=scale,
            is_teacher=is_teacher,
        )

        # --- 아래는 타일 미사용 경로 (요청대로 없애지 않고 주석 처리로 보존) ---
        # img_pad, ph, pw = pad_for_swin(img, window_size, downsample)
        # if use_amp:
        #     with torch.cuda.amp.autocast(dtype=torch.float16):
        #         sr = run_inference(model, diffusion, autoencoder, img_pad, is_teacher=is_teacher)
        # else:
        #     sr = run_inference(model, diffusion, autoencoder, img_pad, is_teacher=is_teacher)
        # sr = remove_padding(sr, ph * scale, pw * scale)
        # ----------------------------------------------------------------

        save_image(sr, out_dir / p.name, normalize_input=normalize_input)

        del img, sr
        torch.cuda.empty_cache()

    print(f"✅ {args.mode.upper()} SR DONE -> {out_dir}")


if __name__ == "__main__":
    main()
