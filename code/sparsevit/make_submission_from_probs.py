#!/usr/bin/env python3
"""Build a 2-column Kaggle submission from saved uint8 probability maps.

Usage:
  python make_submission_from_probs.py PROBS_NPZ THRESHOLD OUT.csv
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def rle_encode(mask: np.ndarray) -> str:
    flat = (np.asarray(mask) > 0.5).astype(np.uint8).reshape(-1, order="F")
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


def main():
    probs_path, threshold, out_path = (Path(sys.argv[1]), float(sys.argv[2]),
                                       Path(sys.argv[3]))
    d = np.load(probs_path)
    probs = d["probs"].astype(np.float32) / 255.0
    ids = [str(s).strip() for s in d["ids"]]
    repo = Path(__file__).resolve().parents[2]
    meta = pd.read_csv(os.environ.get("AIGC_META_CSV", str(repo / "data" / "metadata" / "meta.csv")))
    meta["key"] = meta["ID"].map(lambda i: f"{int(i):06d}")
    meta = meta.set_index("key")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Predicted"])
        for ident, p in zip(ids, probs):
            row = meta.loc[ident]
            h, wi = int(row.orig_h), int(row.orig_w)
            t = torch.from_numpy(p).unsqueeze(0).unsqueeze(0)
            up = F.interpolate(t, size=(h, wi), mode="bilinear",
                               align_corners=False)[0, 0].numpy()
            w.writerow([ident, rle_encode((up > threshold).astype(np.uint8))])
    print(f"wrote {out_path} ({len(ids)} rows, threshold={threshold})")


if __name__ == "__main__":
    main()
