#!/usr/bin/env bash
# Submit a SparseViT prediction to Kaggle and print the resulting score.
# Usage: submit_kaggle.sh [submission.csv]
# Default: 2-column ID,Predicted. The 5-column sample_submission format was
# tried first and Kaggle marked it ERROR; all successful submissions used 2 cols.
set -euo pipefail
source /home/a/Downloads/AIGC_detection/export_kaggle_api_token.sh
export KAGGLE_API_TOKEN
KAGGLE=/tmp/opencode/kagglevenv/bin/kaggle
SLUG=ai-edited-image-localization-cs-5489
FILE="${1:-/home/a/unet/out_final_4500/submission.csv}"
MSG="${2:-SparseViT AAAI2025 official weights, 4500 train, bs8, official sparsity, seed42}"

if [ ! -s "$FILE" ]; then
  echo "submission missing: $FILE" >&2
  exit 1
fi
echo "head of $FILE:"
head -2 "$FILE" | cut -c1-160
echo "rows: $(($(wc -l < "$FILE") - 1))"

$KAGGLE competitions submit -c "$SLUG" -f "$FILE" -m "$MSG"
echo "--- recent submissions ---"
$KAGGLE competitions submissions -c "$SLUG" | head -6
