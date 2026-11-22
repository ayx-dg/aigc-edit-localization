# AIGC Edited-Region Localization — SparseViT vs. the Historical Native Pipeline

Public companion repository for Kaggle competition `ai-edited-image-localization-cs-5489`.
It contains the historical ~0.92 native-resolution code, the current SparseViT
pipeline, the final thesis-driven report (source + PDF), error-localisation data
and images, both best submissions, and a GitHub Action that submits to Kaggle only
when the fixed-name file `submissions/best_submission.csv` is updated and pushed.

## Headline result

| System | Public Score |
|---|---:|
| Historical native-crop U-Net family (best) | **0.92213** |
| SparseViT, whole image square-resized to 512×512 | 0.65764 |
| SparseViT, lossless native 512×512 patches | 0.77279 |
| Weighted two-view ensemble (0.4 global / 0.6 patch, thr 0.59) | **0.80870** |

**Central conclusion:** performance is driven mainly by *pixel-scale fidelity*
and *validation aligned with the official full-image metric*, not by encoder
choice. The remaining errors are concentrated on small, low-contrast edits —
exactly the evidence destroyed by whole-image downscaling. See `report/report.pdf`.

## Repository layout

```
.
├── AGENTS.md                 # repo/data/code guide and usage (Chinese + English)
├── README.md
├── .github/workflows/
│   └── kaggle_submit.yml     # push -> Kaggle submission (secrets required)
├── code/
│   ├── historical_aigc/      # historical 0.90–0.92 native-crop pipeline
│   ├── sparsevit/            # current SparseViT training/inference pipeline
│   │   └── SparseViT/        # upstream AAAI2025 SparseViT source (CC BY 4.0)
│   └── common/rle.py         # verified column-major, 1-based RLE
├── data/
│   ├── metadata/             # split metadata, test IDs, sample submission
│   ├── analysis/             # validation/test error JSON + worst-image CSVs
│   └── error_images/         # per-pixel error/disagreement PNG figures
├── report/
│   ├── report.pdf            # final 13-page English report
│   ├── report.tex            # generated LaTeX source
│   ├── make_report.py        # LaTeX generator
│   ├── figures/              # report figures (PDF)
│   └── *.json                # score and analysis records used by the generator
├── submissions/
│   ├── best_submission.csv            # fixed name watched by the GitHub Action
│   ├── historical_best_0.92213.csv    # archived best historical submission
│   └── sparsevit_v5_0.80870.csv       # archived best SparseViT submission
├── experiments/              # final-run history and metrics
└── scripts/
    ├── analyze_errors.py     # locate worst validation/test images
    ├── make_report_figs.py   # regenerate report figures
    └── submit_kaggle.py      # validate + submit a CSV
```

## Quick start

```bash
# 1. Validate a submission without contacting Kaggle
python scripts/submit_kaggle.py --dry-run

# 2. Rebuild the report from the committed JSON + figures
cd report
python make_report.py
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex

# 3. Re-run the error analysis (needs the external cache/probability artifacts;
#    see AGENTS.md and data/README.md for the environment variables)
AIGC_WORKDIR=/path/to/experiment python scripts/analyze_errors.py
```

## Automatic Kaggle submission on CSV update

`.github/workflows/kaggle_submit.yml` is triggered by a push to `main` **only**
when `submissions/best_submission.csv` changes (plus manual dispatch). It
validates that fixed-name CSV, then submits it with the Kaggle CLI. To submit a
new model, replace `submissions/best_submission.csv` with the new 2-column
`ID,Predicted` CSV and push; other pushes do not trigger a submission.

Add a Kaggle credential secret first (either option works):

```bash
# Preferred: Kaggle API token (KGAT_...)
printf %s "$KAGGLE_API_TOKEN" | gh secret set KAGGLE_API_TOKEN

# Or the classic key pair from kaggle.json:
gh secret set KAGGLE_USERNAME
gh secret set KAGGLE_KEY
```

You must also accept the competition rules once on Kaggle. If the secrets are
missing, the workflow prints a warning and skips the submission instead of
failing. Manual dispatch can override the CSV path and message.

## Data policy

Raw competition images/masks, the 604 MB SparseViT checkpoint, and the saved
`.npz` probability maps are **not** committed (size and competition terms).
The repository includes metadata, analysis JSON, error images, score records,
and submission CSVs — enough to read and regenerate the report and to inspect
the misclassified examples. The upstream SparseViT source is retained under its
CC BY 4.0 license in `code/sparsevit/SparseViT/LICENSE`.

## Key format facts

- Accepted Kaggle submission: two columns `ID,Predicted`; the 5-column
  `sample_submission.csv` layout returns `SubmissionStatus.ERROR`.
- RLE is column-major (Fortran), 1-based `(start, length)`; empty mask = empty string.
- Saved probability maps are `uint8` in `[0,255]`; divide by 255 before thresholding.
- `Score = 0.7*mIoU + 0.3*mRecall`, computed per image then averaged.
