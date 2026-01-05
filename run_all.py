# run_all.py
import subprocess
import sys
import os
from pathlib import Path


def run(cmd, env=None):
    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print("\n❌ ERROR occurred. Stopping pipeline.")
        sys.exit(r.returncode)


def assert_nonempty(dirpath: str):
    d = Path(dirpath)
    if not d.exists():
        raise SystemExit(f"❌ Missing dir: {dirpath}")
    imgs = list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.jpeg"))
    if len(imgs) == 0:
        raise SystemExit(f"❌ No images saved in: {dirpath}")
    print(f"✅ OK: {dirpath} ({len(imgs)} files)")


def main():
    base_env = os.environ.copy()

    # ✅ allocator 안정 설정 (expandable_segments 제거)
    # - max_split_size_mb: fragmentation 완화
    # - garbage_collection_threshold: 캐시 정리 적극화
    base_env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.8"

    # ---- 1) Teacher on GPU0 ----
    env_t = base_env.copy()
    env_t["CUDA_VISIBLE_DEVICES"] = "0"
    run(
        ["python", "test_sr.py", "--mode", "teacher", "--gpu", "0", "--tile", "256", "--overlap", "32"],
        env=env_t
    )

    # ---- 2) Student on GPU1 ----
    env_s = base_env.copy()
    env_s["CUDA_VISIBLE_DEVICES"] = "1"
    # visible GPU가 1개뿐 → 내부 인덱스 0
    run(
        ["python", "test_sr.py", "--mode", "student", "--gpu", "0", "--tile", "256", "--overlap", "32"],
        env=env_s
    )

    assert_nonempty("sr_outputs3/teacher")
    assert_nonempty("sr_outputs3/student")

    # ---- 3) Metrics (CPU only) ----
    env_m = os.environ.copy()
    env_m["CUDA_VISIBLE_DEVICES"] = ""
    run(["python", "test_metric.py"], env=env_m)

    print("\n✅ ALL DONE")


if __name__ == "__main__":
    main()
