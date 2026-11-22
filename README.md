# AIGC Edited-Region Localization — SparseViT vs. the Historical Native Pipeline

Public companion repository for Kaggle competition `ai-edited-image-localization-cs-5489`.
It contains the historical ~0.92 native-resolution code, the current SparseViT
pipeline, the thesis-driven report (source + PDF), error-localisation data and
images, reference submissions, and a repository-owned automation step for Kaggle.

## Headline result

| System | Public Score |
|---|---:|
| Historical native-crop U-Net family (best) | **0.92213** |
| SparseViT, whole image square-resized to 512×512 | 0.65764 |
| SparseViT, lossless native 512×512 patches | 0.77279 |
| Weighted two-view ensemble (0.4 global / 0.6 patch, thr 0.59) | **0.80870** |

**Central conclusion:** performance is driven mainly by *pixel-scale fidelity*
and *validation aligned with the official full-image metric*, not by encoder
choice. The remaining errors concentrate on small, low-contrast edits — exactly
the evidence destroyed by whole-image downscaling. See `report/report.pdf`.

## Who this repository is for

- **Local maintainer** — has the full local cache, probability maps and checkpoints;
  uses them to regenerate analysis artefacts.
- **Contributors / teammates** — no Kaggle account and no repository membership
  required to start. Fork the public repository, use the committed data and
  figures, make changes in a branch, and open a pull request. Codex or any agent
  can work from this repository's `AGENTS.md` without machine-specific setup.
- **Instructors / TAs** — the commands below assume a fresh machine and a dataset
  that has already been provided by the course. No knowledge of the original
  toolchain is assumed; environment setup is explicit.

## Repository layout

```
.
├── AGENTS.md                 # data/code inventory, usage, collaboration rules
├── README.md
├── requirements-report.txt   # minimal deps for report + figures (CPU)
├── requirements.txt          # full deps for training/inference (+ report deps)
├── .github/workflows/
│   └── kaggle_submit.yml     # fixed-CSV handoff to the repository-owned submission step
├── code/
│   ├── historical_aigc/      # historical 0.90–0.92 native-crop pipeline
│   ├── sparsevit/            # current SparseViT training/inference pipeline
│   │   └── SparseViT/        # upstream AAAI2025 source (CC BY 4.0)
│   └── common/rle.py         # verified column-major, 1-based RLE
├── data/
│   ├── metadata/             # split metadata, test IDs, sample submission
│   ├── analysis/             # error JSON + worst-image CSVs
│   └── error_images/         # per-pixel error/disagreement PNGs
├── report/                   # report.pdf, report.tex, generator, figures, score JSON
├── submissions/
│   ├── best_submission.csv            # fixed handoff file; update this to submit
│   ├── candidates/                    # proposed CSVs from pull requests
│   ├── historical_best_0.92213.csv    # archived reference
│   └── sparsevit_v5_0.80870.csv       # archived reference
├── experiments/              # final-run and tuning records
├── scripts/                  # analysis, figure, ensemble and handoff utilities
└── work/                     # local generated data (ignored by git)
```

## Environment on a fresh machine

Python 3.11 is assumed. Install `uv` if it is not already available:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, from the repository root:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install -r requirements-report.txt   # report + figures only
uv pip install -r requirements.txt          # full pipeline (adds torch/timm/smp)
```

Rebuilding the PDF additionally needs `pdflatex` (TeX Live or MiKTeX). Report
regeneration runs on CPU; full training/inference needs a CUDA GPU. All paths in
the repository are relative or configured through `AIGC_*` environment variables.

## Reproduce the report and error evidence (committed data only)

```bash
python report/make_report.py
cd report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex && cd ..
```

The committed JSON, figures and PNG error images are the evidence used by the
report. To regenerate the error analysis itself, the cached images, validation
masks and saved probability maps are required; point the script at them with:

```bash
export AIGC_WORKDIR=/path/to/artifacts
export AIGC_CACHE_DIR="$AIGC_WORKDIR/cache"
export AIGC_TEST_IMAGES=/path/to/provided/dataset/test/images
python scripts/analyze_errors.py
```

## Full pipeline from an already-provided dataset

The dataset is assumed to exist. Two layouts are accepted:

- cache mode: `AIGC_CACHE_DIR` contains `meta.csv`, `imgs/`, `masks/`;
- patch mode: `AIGC_DATA_DIR` contains `train.csv`, `train/images/*.webp`,
  `test/images/*.webp`.

```bash
export AIGC_DATA_DIR=/path/to/provided/dataset
export AIGC_WORKDIR="$PWD/work"
export AIGC_CACHE_DIR="$AIGC_DATA_DIR/cache"
export AIGC_PATCH_DIR="$AIGC_WORKDIR/patches"
export AIGC_SPARSE_CKPT=/path/to/official_sparsevit_checkpoint.pth

python code/sparsevit/make_patches.py --data-dir "$AIGC_DATA_DIR" --out-dir "$AIGC_PATCH_DIR"
python code/sparsevit/train.py --cache-dir "$AIGC_CACHE_DIR" --out-dir "$AIGC_WORKDIR/out_global" \
  --arch sparsevit --encoder-weights "$AIGC_SPARSE_CKPT" --size 512 \
  --batch-size 8 --epochs 10 --lr 1e-4 --weight-decay 1e-4 --drop-path 0.2 --seed 42
python code/sparsevit/train.py --patch-dir "$AIGC_PATCH_DIR" --out-dir "$AIGC_WORKDIR/out_patch" \
  --arch sparsevit --encoder-weights "$AIGC_SPARSE_CKPT" --size 512 \
  --batch-size 8 --epochs 5 --lr 1e-4 --weight-decay 1e-4 --drop-path 0.2 --seed 42
python code/sparsevit/sliding_infer.py --webp-dir "$AIGC_DATA_DIR/test/images" \
  --out "$AIGC_WORKDIR/sw_global" --ckpt "$AIGC_WORKDIR/out_global/best.pt" \
  --mode global --overlap 0.5 --weight hann --save-probs
python code/sparsevit/sliding_infer.py --webp-dir "$AIGC_DATA_DIR/test/images" \
  --out "$AIGC_WORKDIR/sw_native" --ckpt "$AIGC_WORKDIR/out_patch/best.pt" \
  --mode native --overlap 0.5 --weight hann --save-probs
python scripts/make_ensemble_probs.py \
  "$AIGC_WORKDIR/sw_global/test_probs_512.npz" \
  "$AIGC_WORKDIR/sw_native/test_probs_512.npz" \
  "$AIGC_WORKDIR/ensemble.npz" --w-global 0.4 --w-native 0.6
python code/sparsevit/make_submission_from_probs.py "$AIGC_WORKDIR/ensemble.npz" 0.59 \
  "$AIGC_WORKDIR/candidate.csv"
python scripts/submit_kaggle.py --dry-run --submission "$AIGC_WORKDIR/candidate.csv"
```

Thresholds are chosen on validation with `code/sparsevit/sweep_probs.py`; the
values shown above reproduce the best reported configuration.

## Contribution and submission handoff

No Kaggle account or repository write access is needed to contribute:

1. Fork this public repository and clone your fork.
2. Set up the environment as above; run the report or analysis commands.
3. Put any proposed submission CSV in `submissions/candidates/` (two columns
   `ID,Predicted`), or change code/data documentation in a branch.
4. Open a pull request describing the change and the validation Score.

After review and merge, the maintainer copies the chosen candidate to
`submissions/best_submission.csv` **and** updates the companion
`submissions/best_submission.score.json` with its public Score and the CSV's
`sha256`. The fixed CSV name is the handoff point; other pushes do not start
the automation.

### Historical-best gate

The action never treats a new commit as the final score unconditionally. Before
submitting, `scripts/gate_best_submission.py` reads our own Kaggle submission
history at runtime (with `0.92213`, ref `58415875`, as the offline floor) and
compares it with the candidate Score from the sidecar:

- **candidate > historical best** → the CSV is archived, the score registry is
  updated, and the CSV is submitted to Kaggle.
- **candidate ≤ historical best** → nothing is submitted; the action restores
  `best_submission.csv` from the archived best (`submissions/score_registry.json`)
  and rewrites the sidecar, so the repository keeps the historical best entry.

The sidecar `sha256` must match the CSV; a mismatch fails closed. A manual run
can simulate either branch without side effects:

```bash
gh workflow run kaggle_submit.yml -f dry_run=true -f score=0.99
gh workflow run kaggle_submit.yml -f dry_run=true -f score=0.50
gh run view --log    # inspect decision=SUBMIT / decision=KEEP
```

See `AGENTS.md` for the exact collaboration rules.

## Key format facts

- Accepted submission: two columns `ID,Predicted`; the 5-column sample format
  is rejected by Kaggle.
- RLE is column-major (Fortran), 1-based `(start, length)`; empty mask = empty string.
- Saved probability maps are `uint8` in `[0,255]`; divide by 255 before thresholding.
- `Score = 0.7*mIoU + 0.3*mRecall`, computed per image then averaged.

Upstream SparseViT source is retained under CC BY 4.0 in
`code/sparsevit/SparseViT/LICENSE`.
