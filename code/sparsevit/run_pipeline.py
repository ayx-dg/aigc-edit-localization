"""Optional orchestration for the sliding-window and native-patch stages.

The working directory and interpreter are configurable, so the same script can
run on a local GPU box or a remote host. Set AIGC_WORKDIR, AIGC_PYTHON and
AIGC_SPARSE_CKPT before use; all paths default to the repository checkout.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("AIGC_WORKDIR", str(REPO)))
CODE = REPO / "code" / "sparsevit"
PY = os.environ.get("AIGC_PYTHON", sys.executable)
CKPT = os.environ.get("AIGC_CKPT", str(WORK / "out_final_4500" / "best.pt"))
# Official AAAI2025 SparseViT weights; "imagenet" means random init for this arch.
SPARSE_CKPT = os.environ.get("AIGC_SPARSE_CKPT", str(WORK / "SparseViT" / "checkpoint" / "checkpoint.pth"))
FINAL_DONE = WORK / "FINAL_DONE.txt"
LOG = WORK / "pipeline.log"
ERR = WORK / "pipeline.err"
THR = 0.20


def run(cmd, cwd=str(WORK), check=True):
    with LOG.open("a") as fl, ERR.open("a") as fe:
        fl.write(f"\n=== RUN {' '.join(str(x) for x in cmd)} ===\n")
        fl.flush()
        t0 = time.time()
        r = subprocess.run(cmd, cwd=cwd, stdout=fl, stderr=fe)
        fl.write(f"=== END rc={r.returncode} wall={time.time()-t0:.0f}s ===\n")
        fl.flush()
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}")
    return r


def main():
    LOG.write_text("", encoding="utf-8")
    ERR.write_text("", encoding="utf-8")
    with LOG.open("a") as f:
        f.write("=== PIPELINE START ===\n")

    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        if FINAL_DONE.exists():
            break
        time.sleep(120)
    if not FINAL_DONE.exists():
        (WORK / "PIPELINE_FAILED.txt").write_text("final run never finished")
        return
    time.sleep(60)

    jobs = [
        ("val_global", "val_webp", "global"),
        ("val_native", "val_webp", "native"),
        ("test_global", "test_webp", "global"),
        ("test_native", "test_webp", "native"),
    ]
    for name, webp, mode in jobs:
        run([PY, str(CODE / "sliding_infer.py"),
             "--webp-dir", str(WORK / webp),
             "--out", str(WORK / ("sw_" + name)),
             "--ckpt", CKPT, "--mode", mode,
             "--overlap", "0.5", "--weight", "hann",
             "--batch-size", "16", "--threshold", str(THR),
             "--save-probs"])
    (WORK / "SW_DONE.txt").write_text("ok\n", encoding="utf-8")
    with LOG.open("a") as f:
        f.write("=== SW_DONE ===\n")

    run([PY, str(CODE / "train.py"),
         "--patch-dir", str(WORK / "patches"),
         "--out-dir", str(WORK / "out_patch"),
         "--arch", "sparsevit", "--encoder-weights", SPARSE_CKPT,
         "--size", "512", "--batch-size", "8",
         "--epochs", "5", "--lr", "1e-4", "--weight-decay", "1e-4",
         "--drop-path", "0.2", "--seed", "42", "--workers", "2",
         "--save-every-steps", "200"])
    (WORK / "PATCH_TRAIN_DONE.txt").write_text("ok\n", encoding="utf-8")
    with LOG.open("a") as f:
        f.write("=== PIPELINE ALL DONE ===\n")


if __name__ == "__main__":
    main()
