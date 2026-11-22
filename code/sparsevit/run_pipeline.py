"""Automated GPU pipeline on ac1, staged after the final full-data run.

Stage 1 (SW): sliding-window inference on val (global + native) and test
              (global + native) using the current best checkpoint.
Stage 2 (PATCH): train SparseViT from ImageNet weights on native 512px patches.

Everything is logged to C:/Users/Lenovo/unet_full/pipeline.log and writes
SW_DONE.txt / PATCH_TRAIN_DONE.txt markers for the local collector.
"""
import subprocess
import time
from pathlib import Path

OUT = Path("C:/Users/Lenovo/unet_full")
PY = r"C:\Users\Lenovo\.conda\envs\BME6145\python.exe"
CKPT = str(OUT / "out_final_4500" / "best.pt")
# Official AAAI2025 SparseViT pretrained weights (required: "imagenet" = random
# init for this architecture).
SPARSE_CKPT = str(OUT / "SparseViT" / "checkpoint" / "checkpoint.pth")
FINAL_DONE = OUT / "FINAL_DONE.txt"
LOG = OUT / "pipeline.log"
ERR = OUT / "pipeline.err"

# thresholds for the first pass; retuned offline after val analysis
THR = 0.20


def run(cmd, cwd=str(OUT), check=True):
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

    # wait for the final run to release the GPU
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        if FINAL_DONE.exists():
            break
        time.sleep(120)
    if not FINAL_DONE.exists():
        (OUT / "PIPELINE_FAILED.txt").write_text("final run never finished")
        return
    # give the previous process a moment to exit
    time.sleep(60)

    # ---------------- Stage 1: sliding-window inference ----------------
    jobs = [
        ("val_global", "val_webp", "global"),
        ("val_native", "val_webp", "native"),
        ("test_global", "test_webp", "global"),
        ("test_native", "test_webp", "native"),
    ]
    for name, webp, mode in jobs:
        run([PY, str(OUT / "sliding_infer.py"),
             "--webp-dir", str(OUT / webp),
             "--out", str(OUT / ("sw_" + name)),
             "--ckpt", CKPT, "--mode", mode,
             "--overlap", "0.5", "--weight", "hann",
             "--batch-size", "16", "--threshold", str(THR),
             "--save-probs"])
    (OUT / "SW_DONE.txt").write_text("ok\n", encoding="utf-8")
    with LOG.open("a") as f:
        f.write("=== SW_DONE ===\n")

    # ---------------- Stage 2: native-patch training -------------------
    run([PY, str(OUT / "train.py"),
         "--patch-dir", str(OUT / "patches"),
         "--out-dir", str(OUT / "out_patch"),
         "--arch", "sparsevit", "--encoder-weights", SPARSE_CKPT,
         "--size", "512", "--batch-size", "8",
         "--epochs", "5", "--lr", "1e-4", "--weight-decay", "1e-4",
         "--drop-path", "0.2", "--seed", "42", "--workers", "2",
         "--save-every-steps", "200"])
    (OUT / "PATCH_TRAIN_DONE.txt").write_text("ok\n", encoding="utf-8")
    with LOG.open("a") as f:
        f.write("=== PIPELINE ALL DONE ===\n")


if __name__ == "__main__":
    main()
