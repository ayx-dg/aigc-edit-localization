# Current SparseViT pipeline

Training, inference and utility scripts for the AAAI2025 SparseViT experiments.

| File | Purpose |
|---|---|
| `train.py` | Cached or native-patch training; per-epoch metrics and `best.pt` by Score |
| `predict.py` | Validation inference, RLE submission, probability cache |
| `sliding_infer.py` | Full-resolution sliding inference (`--mode global|native|both`) |
| `dataset.py`, `dataset_patch.py`, `make_patches.py` | Data loading and lossless PNG patches |
| `model.py`, `sparsevit_model.py` | Model factory and SparseViT segmentation wrapper |
| `metrics.py`, `sweep_probs.py`, `make_submission_from_probs.py` | Metrics, threshold sweep, submission conversion |
| `analyze_metrics.py`, `run_pipeline.py` | Score analysis and job orchestration |
| `SparseViT/` | Upstream official source (CC BY 4.0, see `SparseViT/LICENSE`) |

Requirements: Python 3.10+, PyTorch, `timm`, `segmentation_models_pytorch`,
`numpy`, `pandas`, `Pillow`, `matplotlib`, `scipy`. See
`SparseViT/requirements.txt` for upstream pins.

Important: SparseViT only supports `--size 512`. Use the official 604 MB
checkpoint as `--encoder-weights`; `imagenet` means random init for this arch.
`predict.py` and `sliding_infer.py` insert `SparseViT/` before the script root in
`sys.path` so the class is imported rather than the package submodule.
