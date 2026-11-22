#!/usr/bin/env python3
"""Locate and visualise misclassified images; compare with the historical 0.92 pipeline.

External inputs (not committed; see data/README.md) are resolved from env vars:
  AIGC_WORKDIR       directory containing cache/ and pipeline_results/
  AIGC_CACHE_DIR     cached 512x512 images/masks/meta.csv
  AIGC_GLOBAL_PROBS  global sliding validation probability NPZ (uint8 0-255)
  AIGC_PATCH_PROBS   native-patch sliding validation probability NPZ
  AIGC_TEST_IMAGES   directory of original test/*.webp images
  AIGC_REPORT_DIR    output directory (default: <repo>/report)

Outputs (under report/):
  error_analysis_val.json   - per-image val scores, worst cases, area-stratified stats
  disagreement_test.json    - per-image old-vs-new mask IoU on the test set
  figures/fig_val_errors.pdf
  figures/fig_test_disagreement.pdf
  figures/fig_area_vs_score.pdf
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
# Where the locally prepared cache/ and pipeline_results/ live. Defaults to the
# repository itself; set AIGC_WORKDIR to the original experiment directory.
ROOT = Path(os.environ.get("AIGC_WORKDIR", str(REPO)))
REPORT = Path(os.environ.get("AIGC_REPORT_DIR", str(REPO / "report")))
FIG = REPORT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("AIGC_CACHE_DIR", str(ROOT / "cache")))
sys.path.insert(0, str(REPO / "code" / "common"))
from rle import rle_decode  # noqa: E402

plt.rcParams.update({"font.size": 7.5, "figure.dpi": 170,
                     "savefig.bbox": "tight"})


def metrics(pred, gt):
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    iou = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    rec = 1.0 if tp + fn == 0 else tp / (tp + fn)
    prec = 1.0 if tp + fp == 0 else tp / (tp + fp)
    return iou, rec, prec, tp, fp, fn


# ------------------------------------------------------------------ validation
meta = pd.read_csv(CACHE / "meta.csv")
val_ids = [f"{int(i):06d}" for i in meta.loc[meta.split == "val", "ID"]]
masks = []
for ident in val_ids:
    m = np.load(CACHE / "masks" / f"{ident}.npy") > 0.5
    if m.shape != (512, 512):
        from torch import from_numpy
        import torch.nn.functional as F
        t = from_numpy(m.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        m = F.interpolate(t, size=(512, 512), mode="nearest")[0, 0].numpy() > 0.5
    masks.append(m)
masks = np.stack(masks)

g = np.load(Path(os.environ.get("AIGC_GLOBAL_PROBS", str(ROOT / "pipeline_results/sw_val_global/test_probs_512.npz"))))["probs"].astype(np.float32) / 255.0
p = np.load(Path(os.environ.get("AIGC_PATCH_PROBS", str(ROOT / "pipeline_results/sw_patch_val_native/test_probs_512.npz"))))["probs"].astype(np.float32) / 255.0
preds = {
    "global_v1": g > 0.20,
    "patch_v3": p > 0.86,
    "ensemble_v5": (0.4 * g + 0.6 * p) > 0.59,
}

val_rows = []
for k, ident in enumerate(val_ids):
    gt_frac = float(masks[k].mean())
    row = {"id": ident, "gt_frac": gt_frac}
    for name, pr in preds.items():
        iou, rec, prec, tp, fp, fn = metrics(pr[k], masks[k])
        row[name] = {"iou": iou, "recall": rec, "precision": prec,
                     "score": 0.7 * iou + 0.3 * rec,
                     "pred_frac": float(pr[k].mean()), "fp": fp, "fn": fn}
    val_rows.append(row)

worst = sorted(val_rows, key=lambda r: r["ensemble_v5"]["score"])[:10]
areas = np.array([r["gt_frac"] for r in val_rows])
score = np.array([r["ensemble_v5"]["score"] for r in val_rows])
bins = np.quantile(areas, [0, .25, .5, .75, 1.0])
area_stats = []
for i in range(4):
    sel = (areas >= bins[i]) & (areas <= bins[i + 1])
    area_stats.append({
        "bin": f"Q{i+1}", "gt_frac_min": float(bins[i]), "gt_frac_max": float(bins[i + 1]),
        "n": int(sel.sum()),
        "mean_score": float(score[sel].mean()),
        "mean_iou": float(np.mean([r["ensemble_v5"]["iou"] for r, s in zip(val_rows, sel) if s])),
    })
val_out = {
    "mean_score": {n: float(np.mean([r[n]["score"] for r in val_rows])) for n in preds},
    "worst_ensemble": worst,
    "area_q": area_stats,
    "corr_area_score": float(np.corrcoef(areas, score)[0, 1]),
}
(REPORT / "error_analysis_val.json").write_text(json.dumps(val_out, indent=2), encoding="utf-8")
print("[val] mean scores", val_out["mean_score"])
print("[val] area quartile scores", [(a["bin"], round(a["mean_score"], 4)) for a in area_stats])
print("[val] corr(area, score)", round(val_out["corr_area_score"], 4))
print("[val] worst", [(w["id"], round(w["ensemble_v5"]["score"], 3),
                       round(w["gt_frac"], 3)) for w in worst[:6]])

# ------------------------------------------------------------------ test disagreement
old_csv = Path(os.environ.get("AIGC_OLD_SUBMISSION", str(REPO / "submissions/historical_best_0.92213.csv")))
new_csv = Path(os.environ.get("AIGC_NEW_SUBMISSION", str(REPO / "submissions/sparsevit_v5_0.80870.csv")))
old = pd.read_csv(old_csv).set_index("ID")["Predicted"].fillna("")
new = pd.read_csv(new_csv).set_index("ID")["Predicted"].fillna("")
meta_t = meta[meta.split == "test"].set_index("ID")
rows = []
for ident in old.index:
    h, w = int(meta_t.loc[ident, "orig_h"]), int(meta_t.loc[ident, "orig_w"])
    mo = rle_decode(old.loc[ident], h, w).astype(bool)
    mn = rle_decode(new.loc[ident], h, w).astype(bool)
    i, rec, prec, tp, fp, fn = metrics(mn, mo)   # treat high-score mask as reference
    rows.append({"id": int(ident), "iou": i, "recall_vs_old": rec,
                 "precision_vs_old": prec,
                 "old_frac": float(mo.mean()), "new_frac": float(mn.mean()),
                 "both_frac": float((mo & mn).sum()) / (h * w),
                 "new_only_frac": float((mn & ~mo).sum()) / (h * w),
                 "old_only_frac": float((mo & ~mn).sum()) / (h * w)})
rows_sorted = sorted(rows, key=lambda r: r["iou"])
dis_out = {
    "old_file": str(old_csv), "new_file": str(new_csv),
    "mean_iou": float(np.mean([r["iou"] for r in rows])),
    "median_iou": float(np.median([r["iou"] for r in rows])),
    "mean_old_frac": float(np.mean([r["old_frac"] for r in rows])),
    "mean_new_frac": float(np.mean([r["new_frac"] for r in rows])),
    "mean_new_only": float(np.mean([r["new_only_frac"] for r in rows])),
    "mean_old_only": float(np.mean([r["old_only_frac"] for r in rows])),
    "worst": rows_sorted[:12],
}
(REPORT / "disagreement_test.json").write_text(json.dumps(dis_out, indent=2), encoding="utf-8")
print("[test] old vs new mean IoU", round(dis_out["mean_iou"], 4))
print("[test] mean fraction old/new", round(dis_out["mean_old_frac"], 4),
      round(dis_out["mean_new_frac"], 4))
print("[test] mean new-only/old-only", round(dis_out["mean_new_only"], 4),
      round(dis_out["mean_old_only"], 4))
print("[test] worst", [(w["id"], round(w["iou"], 3)) for w in rows_sorted[:6]])

# ------------------------------------------------------------------ figures
def overlay(im, mask, color, alpha=0.55):
    arr = np.asarray(im.convert("RGB")).astype(float) / 255.0
    c = np.asarray(color, dtype=float) / 255.0
    m = mask.astype(float)[..., None]
    return np.clip(arr * (1 - alpha * m) + c * alpha * m, 0, 1)


def show_panel(ax, arr, title):
    ax.imshow(arr)
    ax.set_title(title, fontsize=7.5)
    ax.axis("off")


n = 5
fig, axes = plt.subplots(n, 4, figsize=(9.2, 2.3 * n))
for r, w in enumerate(worst[:n]):
    k = val_ids.index(w["id"])
    im = Image.fromarray(np.load(CACHE / "imgs" / f"{w['id']}.npy")).resize((512, 512))
    gt = masks[k]
    show_panel(axes[r, 0], np.asarray(im) / 255.0,
               f"ID {w['id']}  GT {w['gt_frac']:.1%}")
    show_panel(axes[r, 1], overlay(im, gt, (40, 220, 80)),
               f"GT / score {w['ensemble_v5']['score']:.2f}")
    show_panel(axes[r, 2], overlay(im, preds["global_v1"][k], (230, 60, 60)),
               f"global v1  IoU {w['global_v1']['iou']:.2f}")
    show_panel(axes[r, 3], overlay(im, preds["ensemble_v5"][k], (230, 60, 60)),
               f"ensemble v5  IoU {w['ensemble_v5']['iou']:.2f}")
fig.suptitle("Worst validation images for the best SparseViT ensemble (green = ground truth, red = prediction)",
             fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig_val_errors.pdf")
plt.close(fig)

# Mix tiny-edit total disagreements with the largest absolute-area disagreements.
cand = [r for r in rows_sorted[:40] if max(r["old_frac"], r["new_frac"]) > 0.005][:3]
picked_ids = {x["id"] for x in cand}
big = sorted([r for r in rows if r["id"] not in picked_ids],
             key=lambda r: abs(r["new_frac"] - r["old_frac"]), reverse=True)[:2]
vis_rows = cand + big
dis_out["visualised"] = [r["id"] for r in vis_rows]
(REPORT / "disagreement_test.json").write_text(json.dumps(dis_out, indent=2), encoding="utf-8")

test_dir = Path(os.environ.get("AIGC_TEST_IMAGES", str(ROOT / "data/test/images")))
n = len(vis_rows)
fig, axes = plt.subplots(n, 4, figsize=(9.2, 2.3 * n))
for r, w in enumerate(vis_rows):
    ident = f"{w['id']:06d}"
    h, wi = int(meta_t.loc[w["id"], "orig_h"]), int(meta_t.loc[w["id"], "orig_w"])
    im = Image.open(test_dir / f"{ident}.webp").convert("RGB")
    mo = rle_decode(old.loc[w["id"]], h, wi).astype(bool)
    mn = rle_decode(new.loc[w["id"]], h, wi).astype(bool)
    disp = im.resize((min(600, wi), int(im.height * min(600, wi) / wi)))
    scale = disp.width / wi
    mo_s = np.asarray(Image.fromarray((mo * 255).astype(np.uint8)).resize(disp.size, Image.NEAREST)) > 0
    mn_s = np.asarray(Image.fromarray((mn * 255).astype(np.uint8)).resize(disp.size, Image.NEAREST)) > 0
    show_panel(axes[r, 0], np.asarray(disp) / 255.0,
               f"ID {ident}  IoU(old,new)={w['iou']:.2f}")
    show_panel(axes[r, 1], overlay(disp, mo_s, (40, 220, 80)),
               f"historical 0.92  {w['old_frac']:.1%}")
    show_panel(axes[r, 2], overlay(disp, mn_s, (230, 60, 60)),
               f"SparseViT v5  {w['new_frac']:.1%}")
    dis = np.zeros((*mo_s.shape, 3), dtype=float)
    dis[mo_s & mn_s] = [1, 1, 1]
    dis[mn_s & ~mo_s] = [1, 0.2, 0.2]
    dis[mo_s & ~mn_s] = [0.2, 1, 0.2]
    show_panel(axes[r, 3], dis, "red=new only, green=old only")
fig.suptitle("Largest test-set disagreements between the historical 0.92 masks and the SparseViT ensemble",
             fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig_test_disagreement.pdf")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.4))
for name, col in [("global_v1", "#4c72b0"), ("patch_v3", "#8172b2"),
                  ("ensemble_v5", "#55a868")]:
    ax.scatter([r["gt_frac"] for r in val_rows],
               [r[name]["score"] for r in val_rows],
               s=7, alpha=.45, label=name, color=col)
ax.axhline(0.5, color="gray", lw=.6, ls="--")
ax.set_xlabel("Ground-truth edited fraction")
ax.set_ylabel("Per-image Score")
ax.set_title("Small edits are the hard cases (500 validation images)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig_area_vs_score.pdf")
plt.close(fig)

print("figures ->", FIG)
