#!/usr/bin/env python3
"""Generate an English LaTeX report from locally saved experiment results.

Run:  python report/make_report.py
Then: pdflatex -interaction=nonstopmode report.tex   (twice)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
FIG = REPORT / "figures"


def load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def load_hist(p):
    import pandas as pd
    try:
        d = pd.read_csv(p)
        return d if len(d) else None
    except Exception:
        return None


EXP = ROOT / "experiments"
# Prefer the records committed with the repository; fall back to the original
# experiment directory when it is available locally.
final_info = load(EXP / "best_info.json") or load(ROOT / "out_final_4500/best_info.json")
final_hist = load_hist(EXP / "history.csv")
if final_hist is None:
    final_hist = load_hist(ROOT / "out_final_4500/history.csv")
final_cm = load(EXP / "confusion_matrix.json") or load(ROOT / "out_final_4500/confusion_matrix.json")
final_pred = load(EXP / "predict_info.json") or load(ROOT / "out_final_4500/predict_info.json")
kaggle = load(REPORT / "kaggle_score.json")
kaggle_v2 = load(REPORT / "kaggle_score_v2.json")
kaggle_v3 = load(REPORT / "kaggle_score_v3.json")
kaggle_v4 = load(REPORT / "kaggle_score_v4.json")
kaggle_v5 = load(REPORT / "kaggle_score_v5.json")
sw_analysis = load(REPORT / "analysis_sw.json")
all_submissions = [s for s in [kaggle, kaggle_v2, kaggle_v3, kaggle_v4, kaggle_v5] if s]
best_submission = (max(all_submissions, key=lambda s: float(s["score"]))
                   if all_submissions else None)
val_err = load(REPORT / "error_analysis_val.json")
dis = load(REPORT / "disagreement_test.json")
patch_info = (load(EXP / "patch/best_info.json")
              or load(ROOT / "pipeline_results/out_patch/best_info.json"))
historical = [
    ("0.92213", "58415875", "native-crop family, best submission"),
    ("0.92114", "58369543", "native-crop family"),
    ("0.92056", "58373393", "native-crop family"),
    ("0.91484", "58346197", "native-crop family"),
    ("0.90638", "58474337", "EfficientNet-B0 RGB, native 512, epoch 26, single view"),
]

base = (load(EXP / "archs/sparsevit_pretrained/best_info.json")
        or load(ROOT / "out_archs/sparsevit_pretrained/best_info.json"))
tuning = {}
for parent in (EXP / "tuning", ROOT / "out_tuning"):
    if parent.is_dir():
        for d in sorted(parent.iterdir()):
            if d.is_dir() and (d / "best_info.json").exists():
                tuning.setdefault(d.name, load(d / "best_info.json"))
sparse = {}
for parent in (EXP / "sparsity", ROOT / "out_sparsity"):
    if parent.is_dir():
        for d in sorted(parent.iterdir()):
            if d.is_dir() and (d / "best_info.json").exists():
                sparse.setdefault(d.name, load(d / "best_info.json"))

SAMPLES = {"data800": "800/160"}
for n in sparse:
    SAMPLES[n] = "500/100"
SAMPLES["Official (baseline)"] = "500/100"


def best_time(info, name):
    h = None
    if name in tuning:
        h = load_hist(EXP / "tuning" / name / "history.csv")
        if h is None:
            h = load_hist(ROOT / ("out_tuning/" + name + "/history.csv"))
    elif name in sparse:
        h = load_hist(EXP / "sparsity" / name / "history.csv")
        if h is None:
            h = load_hist(ROOT / ("out_sparsity/" + name + "/history.csv"))
    elif name == "Official (baseline)":
        h = load_hist(EXP / "archs/sparsevit_pretrained/history.csv")
        if h is None:
            h = load_hist(ROOT / "out_archs/sparsevit_pretrained/history.csv")
    if h is not None and info:
        try:
            return f"{float(h.iloc[int(info['epoch']) - 1]['elapsed_sec']):.0f}"
        except Exception:
            pass
    return "--"


def row(name, info, s3="8,4,2,1", s4="2,1", batch="4", seed="42"):
    tex_name = name.replace("_", r"\_")
    if not info:
        return f"{tex_name} & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & {seed} \\\\"
    m = info["metrics"]
    return (f"{tex_name} & {s3} & {s4} & {batch} & {SAMPLES.get(name, '500/100')} & "
            f"{m['score']:.4f} & {m['miou']:.4f} & {m['mrecall']:.4f} & "
            f"{m['dice']:.4f} & {m['threshold']:.2f} & {info.get('epoch','--')} & "
            f"{best_time(info, name)} & {seed} \\\\")


# ------------------------------------------------------------------ final run
if final_info and final_hist is not None:
    fm = final_info["metrics"]
    best_ep = int(final_info["epoch"])
    s = final_hist["score"].to_numpy()
    last_drop = s[-1] - s[best_ep - 1]
    val_first, val_last = (float(final_hist.iloc[0]["val_loss"]),
                           float(final_hist.iloc[-1]["val_loss"]))
    train_first, train_last = (float(final_hist.iloc[0]["train_loss"]),
                               float(final_hist.iloc[-1]["train_loss"]))
    ttime = final_info.get("training_time_sec")
    ttime_s = f"{ttime/3600:.2f} h" if ttime else "--"
    ttime_num = f"{ttime:.0f}" if ttime else "--"
    final_row = (f"FINAL-4500 & 8,4,2,1 & 2,1 & 8 & 4500/500 & "
                 f"{fm['score']:.4f} & {fm['miou']:.4f} & {fm['mrecall']:.4f} & "
                 f"{fm['dice']:.4f} & {fm['threshold']:.2f} & {best_ep} & "
                 f"{ttime_num} & {final_info.get('seed', 42)} \\\\")
    final_summary = (
        f"The final full-data run (4500/500, batch 8, official sparsity) reached a best "
        f"\\textbf{{Score of {fm['score']:.4f}}} (mIoU {fm['miou']:.4f}, "
        f"mRecall {fm['mrecall']:.4f}, Dice {fm['dice']:.4f}, threshold "
        f"{fm['threshold']:.2f}) at epoch {best_ep}, with total training time {ttime_s}."
    )
else:
    final_row = ("FINAL-4500 & 8,4,2,1 & 2,1 & 8 & 4500/500 & pending & -- & -- & "
                 "-- & -- & -- & -- & 42 \\\\")
    final_summary = "The final full-data run is still in progress; its numbers will be filled in automatically."

e20 = (load(EXP / "tuning/ep20/best_info.json")
       or load(ROOT / "out_tuning/ep20/best_info.json"))
e20_txt = (
    f"On the 500/100 subset, 20 epochs gave Score {e20['metrics']['score']:.4f}, "
    f"below the 10-epoch baseline of {base['metrics']['score']:.4f}."
    if e20 and base else "The 20-epoch control experiment is not available."
)

if final_cm:
    c = final_cm["counts"]
    cm_txt = (
        f"At the tuned threshold {final_cm['threshold']:.2f}, the pixel-level confusion "
        f"matrix over the validation set is TP={c['tp']:,}, FP={c['fp']:,}, "
        f"FN={c['fn']:,}, TN={c['tn']:,}. Precision={final_cm['precision']:.4f}, "
        f"Recall={final_cm['recall']:.4f}, Specificity={final_cm['specificity']:.4f}."
    )
else:
    cm_txt = "The confusion matrix will be filled in after inference on the final model."

# ------------------------------------------------------------------ build tex
L = []
A = L.append
A(r"\documentclass[11pt]{article}")
A(r"\usepackage{amsmath,graphicx,booktabs,longtable,hyperref}")
A(r"\usepackage[margin=2.2cm]{geometry}")
A(r"\graphicspath{{figures/}}")
A(r"\title{What Drives Performance in AIGC Edited-Region Localization?}")
A(r"\author{SparseViT experiments, error localisation, and comparison with the historical native-resolution pipeline}")
A(r"\date{November 2026}")
A(r"\begin{document}")
A(r"\maketitle")

# ------------------------------------------------------------------ 1. conclusion
A(r"\section{Executive conclusion}")
A(r"\textbf{Claim.} Performance in this task is governed primarily by \emph{pixel-scale "
  r"fidelity} and by \emph{validation aligned with the competition metric}, not by the "
  r"choice of encoder. The historical native-resolution pipeline reaches public scores "
  r"of $0.90$--$0.92$ with ordinary U-Net architectures, while our SparseViT trained on "
  r"square-downscaled whole images reaches only $0.65764$. Preserving original pixels in "
  r"$512\times512$ crops raises SparseViT to $0.77279$, and combining the global and "
  r"native views raises it to $0.80870$. The remaining gap is highly non-uniform: it is "
  r"concentrated on small and low-contrast edits, which are exactly the evidence destroyed "
  r"by whole-image downscaling.")
if kaggle and kaggle_v3 and best_submission:
    A(f"Concretely, the public Score improved from {kaggle['score']} (global SparseViT) "
      f"to {kaggle_v3['score']} (native-patch SparseViT) and then to "
      f"\\textbf{{{best_submission['score']}}} (weighted two-view ensemble). "
      f"On the 500-image validation set, images with ground-truth edited area below $1\\%$ "
      f"score $0.31$--$0.66$, whereas images with more than $5\\%$ edited area score "
      f"$0.87$. On the test set, the historical $0.92$ model and our ensemble agree to "
      f"within a few percent of predicted area on images with substantial edits, but "
      f"differ by up to $72\\times$ in predicted area when the edited region is tiny. "
      f"The practical lesson is therefore architectural in the data sense: keep native "
      f"pixels, select checkpoints with full-image official Score, and use complementary "
      f"views; changing the sparse-attention pattern is a second-order effect.")

# ------------------------------------------------------------------ 2. leaderboard evidence
A(r"\section{Evidence from the public leaderboard}")
A(r"Table~\ref{tab:hist} lists the historical submissions and Table~\ref{tab:ours} our "
  r"submissions. Fig.~\ref{fig:scores} makes the progression visible. The historical "
  r"family was produced by the native-crop code in {\tt code/historical\_aigc/}; the earlier "
  r"downscaled ResNet34 U-Net sanity project produced only an all-edited baseline, not a "
  r"scoring model.")
A(r"\begin{table}[htbp]\centering")
A(r"\caption{Historical public submissions (native-resolution pipeline family).}")
A(r"\label{tab:hist}")
A(r"\begin{tabular}{llp{10cm}}")
A(r"\toprule Score & Ref & Description \\")
A(r"\midrule")
for sc, ref, desc in historical:
    A(f"{sc} & {ref} & {desc} \\\\")
A(r"\bottomrule")
A(r"\end{tabular}")
A(r"\end{table}")
A(r"\begin{table}[htbp]\centering")
A(r"\caption{Our Kaggle submissions on the 500 public test images.}")
A(r"\label{tab:ours}")
A(r"\begin{tabular}{llrr}")
A(r"\toprule Version & Method & Threshold & Public Score \\")
A(r"\midrule")
subs = [("v1", kaggle, "Global 512 cache"),
        ("v2", kaggle_v2, "Global 512 sliding"),
        ("v3", kaggle_v3, "Native 512 patches + sliding"),
        ("v4", kaggle_v4, "Equal-weight ensemble"),
        ("v5", kaggle_v5, "Weighted ensemble (0.4/0.6)")]
subs = [(n, s, d) for n, s, d in subs if s]
for n, s, d in subs:
    A(f"{n} & {d} & {s['threshold']:.2f} & {s['score']} \\\\")
A(r"\bottomrule")
A(r"\end{tabular}")
A(r"\end{table}")
if (FIG / "fig_kaggle_scores.pdf").exists():
    A(r"\begin{figure}[htbp]\centering")
    A(r"\includegraphics[width=.8\linewidth]{fig_kaggle_scores}")
    A(r"\caption{Kaggle public Score by submission.}")
    A(r"\label{fig:scores}")
    A(r"\end{figure}")
A(r"Two negative results are informative. The five-column upload matching "
  r"{\tt sample\_submission.csv} was rejected by Kaggle ({\tt SubmissionStatus.ERROR}), "
  r"so the accepted format is two columns {\tt ID,Predicted}. Submission v2 thresholded "
  r"uint8 probability maps without dividing by 255, effectively predicting almost every "
  r"pixel as edited; this is a pipeline bug, not evidence about the model.")

# ------------------------------------------------------------------ 3. validation errors
A(r"\section{Where the errors are: validation evidence}")
if (FIG / "fig_val_errors.pdf").exists():
    A(r"\begin{figure}[htbp]\centering")
    A(r"\includegraphics[width=.95\linewidth]{fig_val_errors}")
    A(r"\caption{Worst validation images for the best ensemble. Green = ground truth, "
      r"red = prediction. Each row shows the image, the ground-truth overlay, the global "
      r"model, and the ensemble.}")
    A(r"\label{fig:valerr}")
    A(r"\end{figure}")
if val_err:
    A(r"Fig.~\ref{fig:valerr} shows the five lowest-scoring validation images. In every "
      r"case the edited area is small ($0.1\%$--$3.7\%$ of pixels) and the ensemble "
      r"produces an empty or nearly empty mask, so the per-image Score is exactly zero "
      r"because one mask is empty. This is the dominant failure mode of the current "
      r"pipeline: it is not that boundaries are slightly misplaced, but that small edits "
      r"are lost entirely.")
    A(r"\begin{table}[htbp]\centering")
    A(r"\caption{Per-image Score of the best ensemble by ground-truth edited area (500 validation images).}")
    A(r"\begin{tabular}{lrr}")
    A(r"\toprule Edited area & Images & Mean Score \\")
    A(r"\midrule")
    for a in val_err.get("area_fine", []):
        A(f"{a['range'].replace('%', chr(92) + '%')} & {a['n']} & {a['mean_score']:.4f} \\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    A(r"The gradient is steep: $0.31$ for near-empty edits, $0.66$ below $1\%$, and "
      r"$0.87$ above $5\%$. The correlation between edited fraction and Score is only "
      r"$0.19$ because many large-edit images are also difficult, but the area-stratified "
      r"means are unambiguous. Tables~\ref{tab:comp} and the component analysis explain "
      r"why: the global model merges small edited regions into a few large blobs.")
    if "components" in val_err:
        A(r"\begin{table}[htbp]\centering")
        A(r"\caption{Connected-component structure on validation masks (mean per image; "
          r"``tiny'' = components smaller than 20 pixels at $512\times512$).}")
        A(r"\label{tab:comp}")
        A(r"\begin{tabular}{lrrrr}")
        A(r"\toprule Mask & Components & Median size & Tiny comps & Edited fraction \\")
        A(r"\midrule")
        for name, label in [("GT", "Ground truth"), ("global_v1", "Global v1"),
                            ("patch_v3", "Patch v3"), ("ensemble_v5", "Ensemble v5")]:
            c = val_err["components"][name]
            A(f"{label} & {c['components']:.2f} & {c['median_size']:.0f} & "
              f"{c['tiny']:.2f} & {c['frac']:.3f} \\\\")
        A(r"\bottomrule")
        A(r"\end{tabular}")
        A(r"\end{table}")
        A(r"Ground truth contains on average $4.22$ components per image, including "
          r"$1.19$ tiny ones. The global model predicts only $1.83$ components with a "
          r"median size more than twice the ground truth: small edits are absorbed into "
          r"large regions. The native-patch model is much closer ($4.01$ components, "
          r"$1.17$ tiny), and the ensemble is intermediate. This is direct evidence that "
          r"native pixels recover edit structure that downscaling removes.")

# ------------------------------------------------------------------ 4. tradeoff (Figure 3)
A(r"\section{Threshold selection and the mIoU/mRecall trade-off}")
if (FIG / "fig3_miou_recall.pdf").exists():
    A(r"\begin{figure}[htbp]\centering")
    A(r"\includegraphics[width=.85\linewidth]{fig3_miou_recall}")
    A(r"\caption{mIoU versus mRecall across configurations; colour encodes Score.}")
    A(r"\label{fig:tradeoff}")
    A(r"\end{figure}")
A(r"Since $\mathrm{Score}=0.7\,\mathrm{mIoU}+0.3\,\mathrm{mRecall}$, iso-score lines "
  r"have slope $-0.7/0.3\approx-2.33$. The configurations split into two regimes: "
  r"dp0 obtains a high mRecall ($0.818$) but only mIoU $0.456$; bs8 and noamp obtain "
  r"mIoU $0.49$--$0.51$ with slightly lower recall. Maximising recall alone is not "
  r"optimal: predicting everything as positive pushes recall towards 1 but collapses "
  r"mIoU to the edited-area prior. The best configurations lie in the upper-right region "
  r"with high mIoU and undamaged recall. The same logic sets the final threshold: the "
  r"weighted ensemble peaks at $0.59$ on full-image validation, and the historical "
  r"pipeline sweeps the official Score directly instead of using Dice.")

# ------------------------------------------------------------------ 5. test disagreement
A(r"\section{Cross-pipeline disagreement on the test set}")
if dis:
    A(r"Because the test set has no public ground truth, we compare the two best "
      r"submissions directly. Decoding both RLE files gives per-image masks for all "
      r"$500$ test images. The mean mask IoU between the historical $0.92$ model and our "
      f"ensemble is {dis['mean_iou']:.4f} (median {dis['median_iou']:.4f}). Their mean "
      f"predicted edited fractions are {dis['mean_old_frac']:.4f} and "
      f"{dis['mean_new_frac']:.4f}, both close to the training prior $0.1164$, so neither "
      f"model is exploiting a trivial all-positive or all-negative strategy.")
    A(r"\begin{table}[htbp]\centering")
    A(r"\caption{Predicted-area ratio by edited-area bin, using the historical model as reference.}")
    A(r"\begin{tabular}{lrrrr}")
    A(r"\toprule Historical area & Images & Historical mean & New mean & New / old \\")
    A(r"\midrule")
    for lab, n, o, nw in [("$<0.1\\%$", 10, 0.0004, 0.0257),
                          ("$0.1$--$1\\%$", 65, 0.0055, 0.0197),
                          ("$1$--$5\\%$", 154, 0.0269, 0.0272),
                          ("$>5\\%$", 271, 0.1935, 0.1871)]:
        A(f"{lab} & {n} & {o:.4f} & {nw:.4f} & {nw/o:.2f}$\\times$ \\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    A(r"The disagreement is strongly area-dependent: for images where the historical "
      r"model finds less than $0.1\%$ edited pixels, our ensemble predicts $72\times$ "
      r"more area; above $1\%$ the two models agree in total area to within $3\%$. "
      r"The worst per-image IoU values are all tiny-edit cases (historical area "
      r"$0.02\%$--$1.4\%$), where one model predicts an empty mask and the other does not. "
      r"This mirrors the validation finding and locates the residual error precisely.")
    if "components" in dis:
        A(r"The masks also differ in texture. The historical model averages "
          f"{dis['components']['mean_old']:.1f} connected components per image "
          f"({dis['components']['mean_tiny_old']:.1f} smaller than 20 pixels), while our "
          f"ensemble averages {dis['components']['mean_new']:.2f} components "
          f"({dis['components']['mean_tiny_new']:.2f} tiny). The historical masks are far "
          r"more fragmented. Since validation ground truth has $4.22$ components and "
          r"$1.19$ tiny ones, part of this fragmentation is plausibly genuine fine edit "
          r"structure; without test ground truth we cannot separate true small edits from "
          r"false-positive speckles, but the direction of the difference is consistent "
          r"with the recall advantage of the native-crop family.")
    if (FIG / "fig_test_disagreement.pdf").exists():
        A(r"\begin{figure}[htbp]\centering")
        A(r"\includegraphics[width=.95\linewidth]{fig_test_disagreement}")
        A(r"\caption{Largest test-set disagreements. Green = historical $0.92$ model, "
          r"red = SparseViT v5, white = agreement.}")
        A(r"\end{figure}")
    A(r"These figures and tables are the most direct answer to ``which test images are "
      r"wrong'': we cannot know the true labels, but the images with near-empty masks in "
      r"one model and small-but-nonempty masks in the other are the highest-probability "
      r"error cases, and their IDs are stored in {\tt report/disagreement\_test.json}.")

# ------------------------------------------------------------------ 6. mechanism
A(r"\section{Why the historical pipeline wins: a code-level comparison}")
A(r"The two codebases differ in five systematic ways. None of them requires a stronger "
  r"encoder; together they explain the $0.11$ Score gap.")
A(r"\begin{table}[htbp]\centering")
A(r"\caption{Pipeline properties inferred from the two codebases.}")
A(r"\begin{tabular}{p{2.6cm}p{5.6cm}p{5.6cm}}")
A(r"\toprule Property & Historical native pipeline & This SparseViT pipeline \\")
A(r"\midrule")
A(r"Input scale & Original-pixel $512\times512$ crops, no resizing "
  r"({\tt aigc/native.py:1,22-23}) & Whole image resized to square $512\times512$ "
  r"({\tt dataset.py:77-79}); later native crops added \\")
A(r"Crop sampling & Half of crops centred on edited pixels & SparseViT global: none; "
  r"patch stage: positive + hard-negative crops \\")
A(r"Validation & Full-resolution tiled probability maps, exact official Score and "
  r"threshold sweep every epoch ({\tt aigc/native.py:81-108}) & Global: cache-based Score; "
  r"patch stage: patch-level Score, full-image check only afterwards \\")
A(r"Optimisation & AdamW, AMP, gradient clipping, EMA, cosine decay, optional "
  r"teacher--student distillation & AdamW, 10 epochs global / 5 epochs patch, no EMA "
  r"or distillation \\")
A(r"Inference & Overlapping tiles, Hann blending, 4-way flip TTA & Same Hann tiling; "
  r"single view; two-view probability averaging added later \\")
A(r"Architecture & EfficientNet-B0 U-Net or tiny GroupNorm U-Net & SparseViT "
  r"(AAAI2025) with official pretrained weights \\")
A(r"\bottomrule")
A(r"\end{tabular}")
A(r"\end{table}")
A(r"Two observations isolate the mechanism. First, a descriptive historical submission "
  r"explicitly labelled ``EfficientNet-B0 RGB native512 epoch26 single-view baseline'' "
  r"reached $0.90638$: an ordinary U-Net is sufficient once native crops and full-image "
  r"validation are used. Second, our own patch stage preserved pixels and recovered more "
  r"than $0.11$ Score over the global model, but it selected the checkpoint by patch-level "
  r"Score ($0.865$) while the full-image result was only $0.77279$. Metric/target alignment "
  r"is therefore as important as resolution: optimising and selecting on the same "
  r"full-image objective is what allows small edits to survive into the final model.")

# ------------------------------------------------------------------ 7. implications
A(r"\section{Implications and next experiments}")
A(r"\begin{enumerate}")
A(r"\item Train SparseViT with native edit-centred crops \emph{and} evaluate every epoch "
  r"on full images with the official Score, using the existing {\tt validate\_native} "
  r"routine; select the checkpoint by that number, not by patch-level Score.")
A(r"\item Adopt the mature optimisation recipe from the historical pipeline: EMA, cosine "
  r"decay, longer schedules, and flip TTA. These are orthogonal to the encoder.")
A(r"\item Distil from the historical native teacher into SparseViT, or simply fuse the "
  r"two models at probability level; the test-set disagreement analysis shows the views "
  r"carry complementary information, especially on small edits.")
A(r"\item Calibrate the global branch: at threshold $0.20$ it predicts $13.3\%$ edited "
  r"pixels versus $11.7\%$ in the ground truth, whereas the patch model matches the "
  r"prior ($11.8\%$). Recalibrating before averaging should reduce false positives.")
A(r"\end{enumerate}")

# ------------------------------------------------------------------ 8. method record
A(r"\section{Methods and configuration (supporting record)}")
A(r"\subsection{Task and metric}")
A(r"For every image we predict a binary mask (1 = AI-edited pixel, 0 = unedited). "
  r"Metrics are computed per image and then averaged:")
A(r"\begin{equation}")
A(r"\mathrm{IoU}=\frac{TP}{TP+FP+FN},\qquad \mathrm{Recall}=\frac{TP}{TP+FN},"
  r"\qquad \mathrm{Score}=0.7\,\mathrm{mIoU}+0.3\,\mathrm{mRecall}.")
A(r"\end{equation}")
A(r"If both masks are empty, IoU and Recall are 1; if exactly one is empty, both are 0. "
  r"Training uses BCE+Dice loss and the threshold is swept on validation to maximise Score.")
A(r"\subsection{Hyperparameters and terminology}")
A(r"\begin{longtable}{lll}")
A(r"\toprule Term & English & Description / value in this work \\")
A(r"\midrule")
terms = [
    (r"Batch size", r"Batch Size", r"4 or 8 (8 for the final run)"),
    (r"Learning rate", r"Learning Rate", r"$1\times10^{-4}$ with cosine decay"),
    (r"Weight decay", r"Weight Decay", r"$1\times10^{-4}$"),
    (r"Drop path", r"Drop Path", r"SparseViT stochastic depth $=0.2$"),
    (r"Epochs", r"Epochs", r"10 global / 5 patch (one control used 20)"),
    (r"Best epoch", r"Best Epoch", r"Epoch with the highest validation Score"),
    (r"Training samples", r"Train Samples", r"500 / 800 / 4500 global; 9000 patches"),
    (r"Sparsity config", r"Sparsity Config", r"{\tt s\_blocks3} / {\tt s\_blocks4}"),
    (r"Threshold", r"Threshold", r"Chosen by validation Score; $0.59$ for v5"),
    (r"Seed", r"Seed", r"42 for all runs"),
]
for a, b, c in terms:
    A(f"{a} & {b} & {c} \\\\")
A(r"\bottomrule")
A(r"\end{longtable}")
A(r"\subsection{Master experiment table}")
A(r"\begin{longtable}{llrrrrrrrrrrl}")
A(r"\caption{Best results of all SparseViT experiments (checkpoint selected by Score).}\\")
A(r"\toprule")
A(r"Exp. & S3 & S4 & BS & Samples & Score & mIoU & mRecall & Dice & Thr & BestEp & Time & Seed \\")
A(r"\midrule\endfirsthead")
A(r"\toprule Exp. & S3 & S4 & BS & Samples & Score & mIoU & mRecall & Dice & Thr & BestEp & Time & Seed \\")
A(r"\midrule\endhead")
A(r"\bottomrule\endlastfoot")
A(final_row)
A(row("Official (baseline)", base))
for n in sorted(tuning):
    A(row(n, tuning[n], batch="8" if n == "bs8" else "2" if n == "bs2" else "4"))
for n in sorted(sparse):
    s3, s4 = {"sp_coarse": ("4,2,1,1", "1,1"),
               "sp_fine": ("16,8,4,2", "4,2"),
               "sp_vfine": ("32,16,8,4", "8,4"),
               "combo_sp": ("4,2,1,1", "1,1")}.get(n, ("8,4,2,1", "2,1"))
    A(row(n, sparse[n], s3, s4, batch="8" if n == "combo_sp" else "4"))
A(r"\end{longtable}")
A(final_summary)
A(r"\subsection{Sparsity-ratio experiments}")
A(r"\begin{table}[htbp]\centering")
A(r"\caption{Sparsity comparison (batch 4, 500/100, 10 epochs, official pretrained weights unless noted).}")
A(r"\begin{tabular}{llrrrr}")
A(r"\toprule Experiment & S3 & S4 & Batch Size & Samples & Score \\")
A(r"\midrule")
A(r"sp\_coarse & 4,2,1,1 & 1,1 & 4 & 500/100 & 0.5586 \\")
A(r"Official & 8,4,2,1 & 2,1 & 4 & 500/100 & 0.5498 \\")
A(r"sp\_fine & 16,8,4,2 & 4,2 & 4 & 500/100 & 0.5306 \\")
A(r"sp\_vfine & 32,16,8,4 & 8,4 & 4 & 500/100 & 0.5153 \\")
A(r"\bottomrule")
A(r"\end{tabular}")
A(r"\end{table}")
A(r"The official sparsity is close to optimal. Finer tokens consistently hurt because "
  r"the pretrained sparse-coding pattern depends on the original block size; coarser "
  r"blocks gain only $+0.0088$, far below the $+0.0206$ from increasing batch size.")
if patch_info:
    pm = patch_info["metrics"]
    A(r"\subsection{Native-patch training}")
    A(f"Training from official SparseViT weights on {patch_info['train_samples']} lossless "
      f"native patches ({patch_info['val_samples']} validation patches, 5 epochs, batch 8) "
      f"reached patch-level Score {pm['score']:.4f} at epoch {patch_info['epoch']} "
      f"(mIoU {pm['miou']:.4f}, mRecall {pm['mrecall']:.4f}). This number is not directly "
      f"comparable with whole-image validation: a validation sample is a single crop with "
      f"about $20\\%$ edited pixels versus about $12\\%$ for a whole image.")
if sw_analysis:
    s = sw_analysis["validation_sweep"]
    A(r"\subsection{Full-resolution sliding-window inference}")
    A(r"\begin{table}[htbp]\centering")
    A(r"\caption{Validation threshold sweep on saved sliding-window probability maps.}")
    A(r"\begin{tabular}{lrrrr}")
    A(r"\toprule View & Best threshold & mIoU & mRecall & Score \\")
    A(r"\midrule")
    for name, label in [("global", "Global 512"), ("native", "Native sliding"),
                        ("average", "Average of both")]:
        v = s[name]
        A(f"{label} & {v['threshold']:.2f} & {v['miou']:.4f} & "
          f"{v['mrecall']:.4f} & {v['score']:.4f} \\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    A(r"Native sliding with the globally trained model is weak ($0.4765$) because of the "
      r"scale mismatch; the patch-trained model reaches $0.7648$, and averaging the two "
      r"views on whole-image validation reaches $0.7923$ before weight tuning.")

# ------------------------------------------------------------------ 9. reproducibility
A(r"\section{Reproducibility and submission format}")
A(r"The commands below assume a fresh machine and a dataset already supplied by the "
  r"course. They use only relative paths and \texttt{AIGC\_*} variables, so no "
  r"machine-specific setup is required to reproduce the analysis. Python 3.11 and "
  r"\texttt{uv} pin the environment:")
A(r"\begin{verbatim}")
A(r"curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux/macOS")
A(r"# Windows PowerShell: irm https://astral.sh/uv/install.ps1 | iex")
A(r"git clone https://github.com/ayx-dg/aigc-edit-localization.git")
A(r"cd aigc-edit-localization")
A(r"uv venv --python 3.11 .venv && source .venv/bin/activate")
A(r"uv pip install -r requirements-report.txt   # figures and report only")
A(r"uv pip install -r requirements.txt          # full training/inference")
A(r"\end{verbatim}")
A(r"Rebuilding the PDF needs only the committed JSON, figure and PNG evidence plus "
  r"\texttt{pdflatex}; no GPU is involved:")
A(r"\begin{verbatim}")
A(r"python report/make_report.py")
A(r"cd report && pdflatex -interaction=nonstopmode report.tex \\")
A(r"  && pdflatex -interaction=nonstopmode report.tex")
A(r"\end{verbatim}")
A(r"To retrain, set the data and checkpoint locations and run the full pipeline. The "
  r"two sliding-window views are combined at weights $0.4/0.6$ and thresholded at "
  r"$0.59$ to reproduce the best configuration:")
A(r"\begin{verbatim}")
A(r"export AIGC_DATA_DIR=/path/to/provided/dataset")
A(r"export AIGC_WORKDIR=$PWD/work")
A(r"export AIGC_PATCH_DIR=$AIGC_WORKDIR/patches")
A(r"export AIGC_SPARSE_CKPT=/path/to/official_sparsevit_checkpoint.pth")
A(r"python code/sparsevit/make_patches.py --data-dir $AIGC_DATA_DIR --out-dir $AIGC_PATCH_DIR")
A("python code/sparsevit/train.py --cache-dir $AIGC_DATA_DIR/cache --out-dir $AIGC_WORKDIR/out_global \\")
A(r"  --arch sparsevit --encoder-weights $AIGC_SPARSE_CKPT --size 512 \\")
A(r"  --batch-size 8 --epochs 10 --lr 1e-4 --weight-decay 1e-4 --drop-path 0.2 --seed 42")
A("python code/sparsevit/train.py --patch-dir $AIGC_PATCH_DIR --out-dir $AIGC_WORKDIR/out_patch \\")
A(r"  --arch sparsevit --encoder-weights $AIGC_SPARSE_CKPT --size 512 \\")
A(r"  --batch-size 8 --epochs 5 --lr 1e-4 --weight-decay 1e-4 --drop-path 0.2 --seed 42")
A("python code/sparsevit/sliding_infer.py --webp-dir $AIGC_DATA_DIR/test/images \\")
A(r"  --out $AIGC_WORKDIR/sw_global --ckpt $AIGC_WORKDIR/out_global/best.pt \\")
A(r"  --mode global --overlap 0.5 --weight hann --save-probs")
A("python code/sparsevit/sliding_infer.py --webp-dir $AIGC_DATA_DIR/test/images \\")
A(r"  --out $AIGC_WORKDIR/sw_native --ckpt $AIGC_WORKDIR/out_patch/best.pt \\")
A(r"  --mode native --overlap 0.5 --weight hann --save-probs")
A("python scripts/make_ensemble_probs.py $AIGC_WORKDIR/sw_global/test_probs_512.npz \\")
A(r"  $AIGC_WORKDIR/sw_native/test_probs_512.npz $AIGC_WORKDIR/ensemble.npz \\")
A(r"  --w-global 0.4 --w-native 0.6")
A("python code/sparsevit/make_submission_from_probs.py $AIGC_WORKDIR/ensemble.npz 0.59 \\")
A(r"  $AIGC_WORKDIR/candidate.csv")
A(r"\end{verbatim}")
A(r"Predicted probabilities are bilinearly upsampled to the original $H\times W$, "
  r"thresholded, and encoded as column-major (Fortran order), 1-based RLE. A two-column "
  r"file is required:")
A(r"\begin{verbatim}")
A(r"ID,Predicted")
A(r"005001,120 30 500 20")
A(r"005002,")
A(r"\end{verbatim}")
A(r"An empty mask is an empty string. Saved probability maps are uint8 in $[0,255]$ and "
  r"must be divided by 255 before thresholding. Report artefacts, error JSON files and "
  r"archived submissions all live inside the repository. Contributions use forks and "
  r"pull requests: proposed CSVs are placed under \texttt{submissions/candidates/} and "
  r"the maintainer copies the accepted candidate to the fixed handoff file "
  r"\texttt{submissions/best\_submission.csv}; no contributor needs credentials.")

# ------------------------------------------------------------------ appendix
A(r"\appendix")
A(r"\section{Training dynamics and pixel-level errors}")
if (FIG / "fig_curves_final.pdf").exists():
    A(r"\begin{figure}[htbp]\centering")
    A(r"\includegraphics[width=.92\linewidth]{fig_curves_final}")
    A(r"\caption{FINAL-4500 per-epoch train/validation loss, Dice/mIoU/mRecall, Score and threshold.}")
    A(r"\end{figure}")
if (FIG / "fig_convergence_compare.pdf").exists():
    A(r"\begin{figure}[htbp]\centering")
    A(r"\includegraphics[width=.8\linewidth]{fig_convergence_compare}")
    A(r"\caption{Score convergence curves (x-axis: epoch).}")
    A(r"\end{figure}")
if final_hist is not None:
    A(f"Validation loss changes from {val_first:.4f} to {val_last:.4f}; training loss from "
      f"{train_first:.4f} to {train_last:.4f}. The best epoch is {best_ep}, and the last "
      f"epoch differs by {last_drop:+.4f} in Score. " + e20_txt + " With 4500 training "
      f"images, 10 epochs already corresponds to 5620 optimiser steps, so extending to 20 "
      f"offers limited benefit and more overfitting risk.")
if (FIG / "fig_confusion_matrix.pdf").exists():
    A(r"\begin{figure}[htbp]\centering")
    A(r"\includegraphics[width=.55\linewidth]{fig_confusion_matrix}")
    A(r"\caption{Normalised confusion matrix on validation (rows = ground truth).}")
    A(r"\end{figure}")
A(cm_txt)
if (FIG / "fig_area_vs_score.pdf").exists():
    A(r"\begin{figure}[htbp]\centering")
    A(r"\includegraphics[width=.8\linewidth]{fig_area_vs_score}")
    A(r"\caption{Per-image Score versus ground-truth edited fraction for the three SparseViT views.}")
    A(r"\end{figure}")
A(r"The global model predicts $13.3\%$ edited pixels against $11.7\%$ ground truth, "
  r"a systematic over-prediction that lowers mIoU. Lowering the threshold raises recall "
  r"and false positives together; the threshold must be chosen by Score, not by recall.")

A(r"\end{document}")

tex = REPORT / "report.tex"
tex.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"wrote {tex}")
