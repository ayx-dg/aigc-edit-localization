#!/usr/bin/env python3
"""Sweep thresholds on saved validation probability maps.

Compares global / native / averaged sliding-window inference using the exact
competition metric (per-image mIoU/mRecall, then average), and prints the best
threshold for each. Also reports micro metrics for Kaggle-score diagnosis.

Example:
  python sweep_probs.py --probs sw_val_global/test_probs_512.npz \
      sw_val_native/test_probs_512.npz --cache-dir cache
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def per_image_metrics(probs, masks, thr):
    pred = probs > thr
    tp = (pred & masks).sum(axis=(1, 2))
    fp = (pred & ~masks).sum(axis=(1, 2))
    fn = (~pred & masks).sum(axis=(1, 2))
    tn = (~pred & ~masks).sum(axis=(1, 2))
    iou = np.where(tp + fp + fn == 0, 1.0, tp / np.maximum(1, tp + fp + fn))
    rec = np.where(tp + fn == 0, 1.0, tp / np.maximum(1, tp + fn))
    dice = np.where(2 * tp + fp + fn == 0, 1.0,
                    2 * tp / np.maximum(1, 2 * tp + fp + fn))
    return iou.mean(), rec.mean(), dice.mean(), (tp.sum(), fp.sum(), fn.sum(), tn.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probs", nargs="+", type=Path,
                    help="two npz files: global and native (in that order)")
    ap.add_argument("--cache-dir", type=Path, default=Path("/home/a/unet/cache"))
    ap.add_argument("--ids", type=int, nargs="*",
                    help="validate only these ID values (default: all val ids)")
    args = ap.parse_args()

    meta = pd.read_csv(args.cache_dir / "meta.csv")
    val = meta[meta.split == "val"].reset_index(drop=True)
    masks = []
    for _, r in val.iterrows():
        m = np.load(args.cache_dir / "masks" / f"{int(r.ID):06d}.npy") > 0.5
        if m.shape != (512, 512):
            t = torch.from_numpy(m.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            m = F.interpolate(t, size=(512, 512), mode="nearest")[0, 0].numpy() > 0.5
        masks.append(m)
    masks = np.stack(masks)
    print(f"[sweep] {len(masks)} val masks shape={masks.shape[1:]}")

    val_ids = [int(i) for i in val["ID"]]
    arrays = []
    names = []
    for p in args.probs:
        d = np.load(p)
        arr = d["probs"].astype(np.float32) / 255.0
        ids = [int(str(s).strip().lstrip("0") or "0") for s in d["ids"]]
        # Reorder probability maps to exactly match the cache val-ID order.
        pos = {v: k for k, v in enumerate(ids)}
        missing = [v for v in val_ids if v not in pos]
        if missing:
            raise RuntimeError(f"{p.parent.name}: missing val IDs {missing[:5]}")
        arr = arr[[pos[v] for v in val_ids]]
        arrays.append(arr)
        names.append(p.parent.name)
        print(f"[sweep] {p.parent.name}: {arr.shape}")

    if len(arrays) == 2:
        arrays.append(0.5 * (arrays[0] + arrays[1]))
        names.append("average")

    thrs = np.arange(0.02, 0.91, 0.02)
    for arr, name in zip(arrays, names):
        best = None
        print(f"\n--- {name} ---")
        print(f"{'thr':>6} {'mIoU':>8} {'mRecall':>8} "
              f"{'Score':>8} {'microIoU':>9}")
        for thr in thrs:
            miou, mrec, mdice, (tp, fp, fn, tn) = per_image_metrics(arr, masks, thr)
            score = 0.7 * miou + 0.3 * mrec
            miou_micro = tp / max(1, tp + fp + fn)
            print(f"{thr:>6.2f} {miou:>8.4f} {mrec:>8.4f} "
                  f"{score:>8.4f} {miou_micro:>9.4f}")
            if best is None or score > best[1]:
                best = (thr, score, miou, mrec, miou_micro)
        print(f"BEST thr={best[0]:.2f} Score={best[1]:.4f} "
              f"mIoU={best[2]:.4f} mRecall={best[3]:.4f} "
              f"microIoU={best[4]:.4f}")


if __name__ == "__main__":
    main()
