#!/usr/bin/env python3
"""Compare candidate evaluation metrics on validation and print a table.

Reads val_per_image.csv + confusion_matrix.json produced by predict.py.
The Kaggle public score for the same submission is then compared against each
candidate to identify which metric the leaderboard actually uses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--kaggle", type=float, default=None,
                    help="public Kaggle score to compare against")
    args = ap.parse_args()
    d = args.out_dir

    per = pd.read_csv(d / "val_per_image.csv")
    cm = json.loads((d / "confusion_matrix.json").read_text())
    c = cm["counts"]
    TP, FP, FN, TN = c["tp"], c["fp"], c["fn"], c["tn"]

    # ---- per-image metrics (the documented competition metric) ----
    miou = float(per["iou"].replace([np.inf, -np.inf], 1.0).mean())
    mrec = float(per["recall"].replace([np.inf, -np.inf], 1.0).mean())
    mdice = float(per["dice"].replace([np.inf, -np.inf], 1.0).mean())
    mprec = float(per["precision"].replace([np.inf, -np.inf], 1.0).mean())
    score = 0.7 * miou + 0.3 * mrec

    # ---- micro (pooled-pixel) metrics ----
    micro_iou = TP / max(1, TP + FP + FN)
    micro_dice = 2 * TP / max(1, 2 * TP + FP + FN)
    micro_rec = TP / max(1, TP + FN)
    micro_prec = TP / max(1, TP + FP)
    pix_acc = (TP + TN) / max(1, TP + FP + FN + TN)
    micro_f1 = 2 * micro_prec * micro_rec / max(1e-9, micro_prec + micro_rec)

    cands = {
        "per-image mIoU": miou,
        "per-image mRecall": mrec,
        "per-image mDice": mdice,
        "per-image mPrecision": mprec,
        "DOCUMENTED Score = .7mIoU+.3mRecall": score,
        "micro IoU": micro_iou,
        "micro Dice / F1": micro_dice,
        "micro Recall": micro_rec,
        "micro Precision": micro_prec,
        "pixel accuracy": pix_acc,
        ".7*microIoU+.3*microRecall": 0.7 * micro_iou + 0.3 * micro_rec,
        ".7*microDice+.3*microRecall": 0.7 * micro_dice + 0.3 * micro_rec,
        ".5*mIoU+.5*mRecall": 0.5 * miou + 0.5 * mrec,
        "mean(per-image IoU,Recall,F1)": float(np.mean([
            miou, mrec, (2 * mprec * mrec / max(1e-9, mprec + mrec))])),
    }

    print(f"{'metric':<42} {'value':>10}"
          + (f" {'|diff| vs Kaggle':>18}" if args.kaggle else ""))
    print("-" * 74)
    for k, v in sorted(cands.items(), key=lambda kv: -kv[1]):
        extra = f"{abs(v - args.kaggle):>18.4f}" if args.kaggle else ""
        print(f"{k:<42} {v:>10.4f}{extra}")

    if args.kaggle:
        best = min(cands.items(), key=lambda kv: abs(kv[1] - args.kaggle))
        print(f"\nClosest to Kaggle {args.kaggle}: {best[0]} = {best[1]:.4f} "
              f"(diff {abs(best[1]-args.kaggle):.4f})")
        print("TP/FP/FN/TN:", TP, FP, FN, TN)


if __name__ == "__main__":
    main()
