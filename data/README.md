# Data in this repository

## Included

- `metadata/`: Kaggle competition metadata. `meta.csv` gives the 4500/500/500
  train/val/test split, original image size, cached size and edited area.
- `analysis/`: error-analysis JSON and CSV lists of the worst validation images
  and the largest test-set disagreements between the historical 0.92 model and
  SparseViT v5.
- `error_images/`: PNG versions of the per-pixel error/disagreement figures.
  Green = ground truth / historical model, red = SparseViT prediction,
  white = agreement. These are the visual evidence used in the report.

## Not included

The original competition `webp` images, cached 512×512 images/masks, lossless
native patch cache, 604 MB official SparseViT checkpoint and `.npz` probability
maps are intentionally excluded. They are large and subject to the competition
terms. Obtain them from Kaggle and regenerate locally with the scripts in
`code/sparsevit/` and `scripts/`.

## Environment variables for `scripts/analyze_errors.py`

| Variable | Meaning |
|---|---|
| `AIGC_WORKDIR` | experiment root containing `cache/` and `pipeline_results/` |
| `AIGC_CACHE_DIR` | cached images/masks and `meta.csv` |
| `AIGC_GLOBAL_PROBS` | global sliding validation probabilities (uint8 NPZ) |
| `AIGC_PATCH_PROBS` | native-patch sliding validation probabilities (uint8 NPZ) |
| `AIGC_TEST_IMAGES` | directory containing original `test/*.webp` |
| `AIGC_OLD_SUBMISSION` | historical 0.92 submission CSV |
| `AIGC_NEW_SUBMISSION` | SparseViT v5 submission CSV |
| `AIGC_REPORT_DIR` | output directory for JSON and figures (default `../report`) |

Without these variables, the analysis script defaults to paths inside this
repository, where the large external artifacts are absent; use the committed
JSON/PNG files for reading and report regeneration.
