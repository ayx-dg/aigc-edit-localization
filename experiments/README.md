# Experiment records

- `history.csv`: per-epoch train/val loss, Dice, mIoU, mRecall, Score and
  threshold for the final 4500-image SparseViT global run.
- `best_info.json`, `predict_info.json`: best epoch/metrics and prediction
  configuration for that run.
- `confusion_matrix.json`: pixel-level confusion matrix on validation.

Aggregate Kaggle scores and threshold sweeps are in `../report/kaggle_score*.json`
and `../report/analysis*.json`; the thesis-driven report is `../report/report.pdf`.
