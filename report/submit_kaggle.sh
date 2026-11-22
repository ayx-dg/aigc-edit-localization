#!/usr/bin/env bash
# Backward-compatible wrapper. The portable implementation lives in scripts/.
set -euo pipefail
exec python "$(dirname "$0")/../scripts/submit_kaggle.py" "$@"
