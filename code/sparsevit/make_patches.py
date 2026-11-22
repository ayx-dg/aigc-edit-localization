#!/usr/bin/env python3
"""Build a native-resolution 512x512 patch dataset from the full-res WebP images.

Lossless pipeline: WebP decode -> uint8 RGB; RLE decode -> uint8 mask;
spatial crop -> PNG/uint8. No resizing of the input image is ever performed,
so small edited regions and compression-artifact boundaries are preserved.

For each image we sample:
  * one (or more) patches centred on edited pixels,
  * one hard-negative patch (a window with < --neg-frac edited pixels).

Output layout under --out-dir:
    images/<ID>_<k>.png    512x512x3 uint8
    masks/<ID>_<k>.png     512x512   uint8 (0 / 255)
    manifest.csv           id,kind,cy,cx,edited_frac,src_h,src_w

Run locally (CPU only):
    python code/sparsevit/make_patches.py --data-dir DATA_DIR --out-dir WORK_DIR/patches
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code" / "common"))
from rle import rle_decode  # noqa: E402  (verified column-major, 1-based)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path,
                   default=Path(os.environ.get("AIGC_DATA_DIR", str(REPO / "data" / "raw"))))
    p.add_argument("--out-dir", type=Path,
                   default=Path(os.environ.get("AIGC_PATCH_DIR", str(REPO / "work" / "patches"))))
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--per-image", type=int, default=2,
                   help="total patches per image (>=2: 1 positive + rest negative)")
    p.add_argument("--neg-frac", type=float, default=0.01,
                   help="max edited fraction allowed in a hard-negative patch")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="only first N images (test)")
    p.add_argument("--split", choices=["train", "val", "all"], default="all",
                   help="use a fixed 4500/500 ID split (default: all 5000)")
    return p.parse_args()


# The training run uses 4500 train / 500 val with seed 42. Recreate that split
# so a patch model sees exactly the same images as the current best model.
def make_split(n, seed=42, n_val=500):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    val = set((idx[:n_val] + 1).tolist())
    return val


def crop_window(arr, cy, cx, half):
    """Crop a 2*half square centred at (cy,cx), clamping to the image border."""
    h, w = arr.shape[:2]
    y0 = int(np.clip(cy - half, 0, h - 2 * half))
    x0 = int(np.clip(cx - half, 0, w - 2 * half))
    return arr[y0:y0 + 2 * half, x0:x0 + 2 * half]


def process_one(job):
    """Worker: decode one image, sample patches, write PNGs. Returns rows."""
    (id_, img_path, ann, height, width, ps, n_patches, neg_frac,
     out_images, out_masks, rng_seed) = job

    rng = np.random.default_rng(rng_seed)
    half = ps // 2
    name = f"{int(id_):06d}"

    # ---- lossless decode ----
    with Image.open(img_path) as im:
        image = np.asarray(im.convert("RGB"), dtype=np.uint8)
    mask = rle_decode(ann, height, width)  # uint8 (H,W), 0/1

    if image.shape[:2] != (height, width):
        return [(name, -1, "ERROR", 0, 0, -1.0, height, width,
                 f"shape {image.shape[:2]} != {(height, width)}")]

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return [(name, -1, "EMPTY", 0, 0, 0.0, height, width, "SKIP")]

    rows = []
    n_pos = max(1, n_patches - 1)
    n_neg = n_patches - n_pos

    # ---- positive patches: centres drawn from edited pixels ----
    # Spread them out when there are several, by rejecting centres too close
    # to an already chosen one.
    chosen = []
    for _ in range(n_pos):
        best, best_d = None, -1.0
        for _try in range(30):
            k = int(rng.integers(0, ys.size))
            cy, cx = int(ys[k]), int(xs[k])
            d = min(((cy - c[0]) ** 2 + (cx - c[1]) ** 2) ** 0.5
                    for c in chosen) if chosen else 1e9
            if d > best_d:
                best, best_d = (cy, cx), d
            if d > ps * 0.6:
                break
        chosen.append(best)

    # ---- hard negatives: random centres with few edited pixels ----
    negs = []
    for _ in range(n_neg):
        found = None
        for _try in range(40):
            cy = int(rng.integers(half, max(half + 1, height - half)))
            cx = int(rng.integers(half, max(half + 1, width - half)))
            m = crop_window(mask, cy, cx, half)
            if m.mean() < neg_frac:
                found = (cy, cx)
                break
        if found is None:
            # fall back to the least-edited of 10 random windows
            cand = []
            for _try in range(10):
                cy = int(rng.integers(half, max(half + 1, height - half)))
                cx = int(rng.integers(half, max(half + 1, width - half)))
                m = crop_window(mask, cy, cx, half)
                cand.append((m.mean(), (cy, cx)))
            found = min(cand)[1]
        negs.append(found)

    # ---- write ----
    for k, ((cy, cx), kind) in enumerate(
            [(c, "pos") for c in chosen] + [(c, "neg") for c in negs]):
        ip = crop_window(image, cy, cx, half)
        mp = crop_window(mask, cy, cx, half)
        stem = f"{name}_{k}"
        Image.fromarray(ip, mode="RGB").save(out_images / f"{stem}.png",
                                              compress_level=3)
        Image.fromarray((mp * 255).astype(np.uint8), mode="L").save(
            out_masks / f"{stem}.png", compress_level=6)
        rows.append((name, k, kind, cy, cx, float(mp.mean()), height, width, "OK"))
    return rows


def main():
    args = parse_args()
    out_images = args.out_dir / "images"
    out_masks = args.out_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.read_csv(args.data_dir / "train.csv")
    df = df[["ID", "annotation", "height", "width"]].copy()
    df["annotation"] = df["annotation"].fillna("").astype(str)

    if args.split in ("train", "val"):
        # Reuse the exact 4500/500 split of the current best model so results
        # are directly comparable and the validation set stays unseen.
        meta = pd.read_csv(os.environ.get("AIGC_META_CSV", str(REPO / "data" / "metadata" / "meta.csv")))
        val_ids = set(meta.loc[meta["split"] == "val", "ID"].astype(int).tolist())
        if args.split == "train":
            df = df[~df["ID"].isin(val_ids)]
        else:
            df = df[df["ID"].isin(val_ids)]
        df = df.reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)

    jobs = []
    for _, r in df.iterrows():
        id_ = int(r["ID"])
        img_path = args.data_dir / "train" / "images" / f"{id_:06d}.webp"
        jobs.append((id_, img_path, r["annotation"], int(r["height"]),
                     int(r["width"]), args.patch_size, args.per_image,
                     args.neg_frac, out_images, out_masks,
                     args.seed * 1000003 + id_))

    print(f"[patches] split={args.split} images={len(jobs)} "
          f"per_image={args.per_image} size={args.patch_size} "
          f"out={args.out_dir}", flush=True)
    t0 = time.time()
    all_rows = []
    errors = []
    with Pool(args.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(process_one, jobs, chunksize=4)):
            for row in rows:
                (all_rows if row[8] == "OK" else errors).append(row)
            if (i + 1) % 500 == 0 or i + 1 == len(jobs):
                print(f"[patches] {i+1}/{len(jobs)} images, "
                      f"{len(all_rows)} patches, {time.time()-t0:.0f}s", flush=True)

    man = args.out_dir / f"manifest_{args.split}.csv"
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "k", "kind", "cy", "cx", "edited_frac",
                    "src_h", "src_w", "status"])
        w.writerows(all_rows + errors)
    n_pos = sum(r[1] == "pos" for r in all_rows)
    n_neg = sum(r[1] == "neg" for r in all_rows)
    print(f"[patches] DONE {len(all_rows)} patches "
          f"(pos={n_pos}, neg={n_neg}), errors={len(errors)}, "
          f"time={time.time()-t0:.0f}s -> {man}", flush=True)


if __name__ == "__main__":
    main()
