#!/usr/bin/env python3
"""Generate report figures + analysis text from all local experiment results.

Re-run after changing experiment results (set AIGC_WORKDIR to the experiment root):
    python scripts/make_report_figs.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("AIGC_WORKDIR", str(Path(__file__).resolve().parents[1])))
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 160, "savefig.bbox": "tight",
})


def load_history(path):
    if not path.exists():
        return None
    try:
        d = pd.read_csv(path)
        return d if len(d) else None
    except Exception:
        return None


def load_info(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------- experiments
EXPS = {}
for folder, prefix in ((ROOT / "out_tuning", "tune/"),
                       (ROOT / "out_sparsity", "")):
    if not folder.exists():
        continue
    for d in sorted(folder.iterdir()):
        if not d.is_dir():
            continue
        h = load_history(d / "history.csv")
        i = load_info(d / "best_info.json")
        if h is not None and i is not None:
            EXPS[prefix + d.name] = (h, i, d)

base_info = load_info(ROOT / "out_archs/sparsevit_pretrained/best_info.json")
base_hist = load_history(ROOT / "out_archs/sparsevit_pretrained/history.csv")
if base_info and base_hist is not None:
    EXPS["Official(baseline)"] = (base_hist, base_info,
                                  ROOT / "out_archs/sparsevit_pretrained")
final_dir = ROOT / "out_final_4500"
final_hist = load_history(final_dir / "history.csv")
final_info = load_info(final_dir / "best_info.json")
if final_hist is not None and final_info:
    EXPS["FINAL-4500"] = (final_hist, final_info, final_dir)

analysis = {}

# ---------------------------------------------------------------- Fig 3
if EXPS:
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    cmap = plt.get_cmap("viridis")
    scores = [i["metrics"]["score"] for _, i, _ in EXPS.values()]
    lo, hi = min(scores), max(scores)
    for (name, (_, info, _)), s in zip(EXPS.items(), scores):
        m = info["metrics"]
        c = cmap((s - lo) / max(1e-9, hi - lo))
        ax.scatter(m["miou"], m["mrecall"], s=85 if name == "FINAL-4500" else 50,
                   color=c, edgecolor="k", linewidth=0.5, zorder=3)
        if name in ("FINAL-4500", "Official(baseline)", "tune/bs8", "tune/dp0",
                    "sp_coarse", "sp_fine", "sp_vfine", "tune/lr2e4"):
            ax.annotate(name, (m["miou"], m["mrecall"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                      norm=plt.Normalize(lo, hi)), ax=ax)
    cb.set_label("Score = 0.7 mIoU + 0.3 mRecall")
    ax.set_xlabel("mIoU (region overlap)")
    ax.set_ylabel("mRecall (edited-region recall)")
    ax.set_title("mIoU vs mRecall trade-off across configurations")
    fig.savefig(FIG / "fig3_miou_recall.pdf")
    plt.close(fig)

# ---------------------------------------------------------------- final curves
if final_hist is not None:
    ep = final_hist["epoch"].to_numpy()
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.6))
    axs[0, 0].plot(ep, final_hist["train_loss"], "o-", label="train loss")
    axs[0, 0].plot(ep, final_hist["val_loss"], "s-", label="val loss")
    axs[0, 0].set_title("Loss"); axs[0, 0].legend()
    axs[0, 1].plot(ep, final_hist["val_dice"], "o-", label="Dice")
    axs[0, 1].plot(ep, final_hist["miou"], "s-", label="mIoU")
    axs[0, 1].plot(ep, final_hist["mrecall"], "^-", label="mRecall")
    axs[0, 1].set_title("Validation metrics"); axs[0, 1].legend()
    axs[1, 0].plot(ep, final_hist["score"], "o-", color="crimson", label="Score")
    best_ep = int(final_info["epoch"])
    axs[1, 0].axvline(best_ep, ls="--", color="gray", label=f"best ep={best_ep}")
    axs[1, 0].set_title("Competition Score"); axs[1, 0].legend()
    axs[1, 1].plot(ep, final_hist["threshold"], "o-", label="tuned threshold")
    axs[1, 1].set_title("Binarisation threshold"); axs[1, 1].legend()
    for ax in axs.ravel():
        ax.set_xlabel("epoch")
    fig.suptitle("FINAL-4500 training dynamics (bs8, 4500 train, official sparsity)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_curves_final.pdf")
    plt.close(fig)

    # numeric analysis
    s = final_hist["score"].to_numpy()
    vl = final_hist["val_loss"].to_numpy()
    tl = final_hist["train_loss"].to_numpy()
    analysis["final"] = {
        "best_epoch": best_ep,
        "best_score": float(final_info["best_value"]),
        "last_score": float(s[-1]),
        "score_drop_from_best": float(s[-1] - s[best_ep - 1]),
        "val_loss_first": float(vl[0]), "val_loss_last": float(vl[-1]),
        "train_loss_first": float(tl[0]), "train_loss_last": float(tl[-1]),
        "time_sec": final_info.get("training_time_sec"),
        "seed": final_info.get("seed"),
        "threshold": float(final_info["metrics"]["threshold"]),
    }

# ---------------------------------------------------------------- compare curves
names = ["Official(baseline)", "tune/bs8", "sp_coarse", "FINAL-4500"]
fig, ax = plt.subplots(figsize=(6.8, 4.2))
for name in names:
    if name not in EXPS:
        continue
    h = EXPS[name][0]
    ax.plot(h["epoch"], h["score"], "o-", ms=3.5, label=name)
ax.axhline(0.5498, ls=":", color="gray", lw=1)
ax.set_xlabel("epoch"); ax.set_ylabel("Score")
ax.set_title("Convergence speed across configurations")
ax.legend(fontsize=7)
fig.savefig(FIG / "fig_convergence_compare.pdf")
plt.close(fig)

# ---------------------------------------------------------------- sparsity bars
sp = [n for n in EXPS if n in ("sp_coarse", "Official(baseline)", "sp_fine",
                               "sp_vfine", "combo_sp")]
if sp:
    order = ["sp_coarse", "Official(baseline)", "sp_fine", "sp_vfine", "combo_sp"]
    order = [n for n in order if n in sp]
    vals = [EXPS[n][1]["metrics"]["score"] for n in order]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bars = ax.bar(range(len(order)), vals,
                  color=["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#937860"])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Sparsity configuration comparison (500/100, 10 epochs)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sparsity_bars.pdf")
    plt.close(fig)

# ---------------------------------------------------------------- confusion matrix
cm_path = final_dir / "confusion_matrix.json"
if cm_path.exists():
    cm = json.loads(cm_path.read_text(encoding="utf-8"))
    arr = np.array(cm["normalized"])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(arr, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticklabels(["true 0", "true 1"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    ax.set_title(f"Confusion matrix (norm. by row, thr={cm['threshold']:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center",
                    color="white" if arr[i, j] > 0.5 else "black", fontsize=11)
    fig.colorbar(im, ax=ax)
    fig.savefig(FIG / "fig_confusion_matrix.pdf")
    plt.close(fig)
    analysis["confusion"] = cm

# ---------------------------------------------------------------- Kaggle submissions
scores = []
for name, f in [("v1 global", "kaggle_score.json"),
                ("v2 global SW", "kaggle_score_v2.json"),
                ("v3 patch native", "kaggle_score_v3.json"),
                ("v4 ensemble", "kaggle_score_v4.json"),
                ("v5 weighted", "kaggle_score_v5.json")]:
    p = ROOT / "report" / f
    if p.exists():
        info = json.loads(p.read_text(encoding="utf-8"))
        scores.append((name, float(info["score"])))
if scores:
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    names = [s[0] for s in scores]
    vals = [s[1] for s in scores]
    colors = ["#c44e52" if v == min(vals) else "#55a868" if v == max(vals)
              else "#4c72b0" for v in vals]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("Kaggle public Score")
    ax.set_title("Kaggle public score by submission")
    ax.set_ylim(0, max(vals) * 1.18)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.5f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_kaggle_scores.pdf")
    plt.close(fig)
    analysis["kaggle_submissions"] = dict(scores)

# ---------------------------------------------------------------- ep20 evidence
if "tune/ep20" in EXPS and "Official(baseline)" in EXPS:
    analysis["ep20"] = {
        "ep20_score": EXPS["tune/ep20"][1]["metrics"]["score"],
        "baseline_score": EXPS["Official(baseline)"][1]["metrics"]["score"],
    }

out = ROOT / "report" / "analysis.json"
out.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"figures -> {FIG}")
print(f"analysis -> {out}")
print(json.dumps(analysis, indent=2, ensure_ascii=False))
