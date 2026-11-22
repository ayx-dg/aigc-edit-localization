#!/usr/bin/env python3
"""Native-resolution sliding-window inference for the SparseViT checkpoint.

Runs on ac1 (GPU). Reads lossless WebP images, tiles them into 512x512
windows with overlap, runs the model, blends the overlapping probability maps,
encodes the result as RLE and writes a Kaggle submission.

Modes
-----
  global : downscale the whole image to 512x512 (the model's training view).
           Reproduces the current predict.py, but reading full-res WebP.
  native : sliding 512x512 windows at native resolution + overlap blending.
  both   : average of global and native probability maps.

Example (ac1):
  python sliding_infer.py --webp-dir test_webp --out out_sw_native \
      --ckpt out_final_4500/best.pt --mode native --overlap 0.5 --batch-size 16
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
# Keep the official SparseViT directory ahead of ROOT: ROOT/SparseViT is a
# package, so if ROOT wins, `from SparseViT import SparseViT` binds a module
# instead of the class and SparseViT_Mul fails with "module is not callable".
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SparseViT"))
from model import build_model  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--webp-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--mode", choices=["global", "native", "both"], default="native")
    p.add_argument("--overlap", type=float, default=0.5,
                   help="fraction of window overlap for native mode")
    p.add_argument("--weight", choices=["constant", "hann"], default="hann")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save-probs", action="store_true",
                   help="save downsampled probability maps for ensembling")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def rle_encode(mask: np.ndarray) -> str:
    """Column-major (Fortran), 1-based (start, length) RLE."""
    binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    flat = binary.reshape(-1, order="F")
    if not flat.any():
        return ""
    padded = np.concatenate(([0], flat, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    lengths = ends - starts + 1
    out = np.empty(2 * len(starts), dtype=int)
    out[0::2], out[1::2] = starts, lengths
    return " ".join(map(str, out.tolist()))


@torch.no_grad()
def infer_windows(model, patches, batch_size, device, use_amp):
    """patches: (N,3,S,S) uint8 -> (N,S,S) float32 probabilities."""
    outs = []
    mean = torch.as_tensor(MEAN, device=device).view(3, 1, 1)
    std = torch.as_tensor(STD, device=device).view(3, 1, 1)
    for i in range(0, len(patches), batch_size):
        xb = torch.from_numpy(patches[i:i + batch_size]).to(
            device, dtype=torch.float32, non_blocking=True) / 255.0
        xb = xb.permute(0, 3, 1, 2)          # HWC -> CHW
        xb = (xb - mean) / std
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xb)
        outs.append(torch.sigmoid(logits.float())[:, 0].cpu().numpy())
    return np.concatenate(outs, axis=0)


def native_prob(model, image, size, stride, wmode, batch_size, device, use_amp):
    """Sliding-window probability map for one (H,W,3) uint8 image."""
    h, w = image.shape[:2]
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    # Reflect pad so border windows have real image content.
    imp = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    H, W = imp.shape[:2]

    ys = list(range(0, H - size + 1, stride))
    xs = list(range(0, W - size + 1, stride))
    if ys[-1] != H - size:
        ys.append(H - size)
    if xs[-1] != W - size:
        xs.append(W - size)

    patches = np.stack([imp[y:y + size, x:x + size] for y in ys for x in xs])
    probs = infer_windows(model, patches, batch_size, device, use_amp)

    if wmode == "hann":
        v = np.hanning(size).astype(np.float32)
        wmap = np.outer(v, v)
    else:
        wmap = np.ones((size, size), dtype=np.float32)

    acc = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    k = 0
    for y in ys:
        for x in xs:
            acc[y:y + size, x:x + size] += probs[k] * wmap
            wsum[y:y + size, x:x + size] += wmap
            k += 1
    return (acc / np.maximum(wsum, 1e-6))[:h, :w]


def global_prob(model, image, size, batch_size, device, use_amp):
    h, w = image.shape[:2]
    small = np.asarray(Image.fromarray(image).resize((size, size),
                                                      Image.BILINEAR))
    p = infer_windows(model, small[None], batch_size, device, use_amp)[0]
    return np.asarray(Image.fromarray((p * 255).astype(np.uint8)).resize(
        (w, h), Image.BILINEAR), dtype=np.float32) / 255.0


def main():
    args = parse_args()
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    args.out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ta = ck.get("args", {})
    torch.manual_seed(int(ta.get("seed", 42)))
    model = build_model(ta.get("encoder_weights", ""), "sparsevit",
                        img_size=int(ta.get("size", args.size)),
                        drop_path=float(ta.get("drop_path", 0.2)),
                        s_blocks3=ta.get("s_blocks3"),
                        s_blocks4=ta.get("s_blocks4")).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    del ck
    stride = int(round(args.size * (1.0 - args.overlap)))
    print(f"[sw] mode={args.mode} size={args.size} stride={stride} "
          f"weight={args.weight} thr={args.threshold} device={device} "
          f"amp={use_amp}", flush=True)

    files = sorted(args.webp_dir.glob("*.webp"))
    if args.limit:
        files = files[:args.limit]
    print(f"[sw] {len(files)} images in {args.webp_dir}", flush=True)

    t0 = time.time()
    rows, rows5, small_probs, empty = [], [], [], 0
    for i, fp in enumerate(files):
        with Image.open(fp) as im:
            image = np.asarray(im.convert("RGB"), dtype=np.uint8)
        h, w = image.shape[:2]

        if args.mode == "global":
            prob = global_prob(model, image, args.size, args.batch_size,
                               device, use_amp)
        elif args.mode == "native":
            prob = native_prob(model, image, args.size, stride, args.weight,
                               args.batch_size, device, use_amp)
        else:
            pg = global_prob(model, image, args.size, args.batch_size,
                             device, use_amp)
            pn = native_prob(model, image, args.size, stride, args.weight,
                             args.batch_size, device, use_amp)
            prob = 0.5 * (pg + pn)

        mask = prob > args.threshold
        rle = rle_encode(mask.astype(np.uint8))
        if not rle:
            empty += 1
        ident = fp.stem
        rows.append({"ID": ident, "Predicted": rle})
        rows5.append({"ID": ident, "annotation": rle,
                      "height": h, "width": w, "Usage": "Public"})
        if args.save_probs:
            psmall = np.asarray(Image.fromarray((prob * 255).astype(np.uint8))
                                .resize((512, 512), Image.BILINEAR),
                                dtype=np.uint8)
            small_probs.append(psmall)
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            print(f"[sw] {i+1}/{len(files)} {fp.name} {h}x{w} "
                  f"pos={mask.mean():.4f} {time.time()-t0:.0f}s", flush=True)

    with (args.out / "submission.csv").open("w", newline="",
                                            encoding="utf-8") as f:
        wtr = csv.DictWriter(f, ["ID", "Predicted"])
        wtr.writeheader(); wtr.writerows(rows)
    # 5-column format matching sample_submission.csv (used by the original
    # project that produced the 0.90+ public submissions).
    with (args.out / "submission_sample_format.csv").open(
            "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, ["ID", "annotation", "height", "width", "Usage"])
        wtr.writeheader(); wtr.writerows(rows5)
    if args.save_probs:
        np.savez_compressed(args.out / "test_probs_512.npz",
                            ids=np.array([r["ID"] for r in rows]),
                            probs=np.stack(small_probs))
    (args.out / "sw_info.json").write_text(json.dumps({
        "mode": args.mode, "overlap": args.overlap, "weight": args.weight,
        "threshold": args.threshold, "size": args.size, "stride": stride,
        "checkpoint": str(args.ckpt), "images": len(rows), "empty": empty,
        "seconds": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")
    print(f"[sw] DONE {len(rows)} rows, {empty} empty, "
          f"{time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
