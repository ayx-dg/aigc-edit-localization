#!/usr/bin/env python3
"""Validate and submit a CSV to the Kaggle AIGC edit-localization competition.

The accepted submission format is **two columns**: ``ID,Predicted``.
The 5-column ``sample_submission.csv`` layout was tested and Kaggle returned
``SubmissionStatus.ERROR``.

Usage:
    python scripts/submit_kaggle.py --dry-run
    KAGGLE_USERNAME=... KAGGLE_KEY=... python scripts/submit_kaggle.py

Credentials are read from the ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``
environment variables (GitHub Actions secrets with the same names).
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SUBMISSION = REPO / "submissions" / "best_submission.csv"
DEFAULT_SLUG = "ai-edited-image-localization-cs-5489"
DEFAULT_MESSAGE = "SparseViT v5 weighted ensemble (w_global=0.4, w_patch=0.6), thr=0.59"


def validate_submission(path: Path) -> int:
    """Check the header and count rows; return the number of data rows."""
    csv.field_size_limit(sys.maxsize)  # RLE strings can exceed 128 KiB
    if not path.is_file():
        raise SystemExit(f"submission not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise SystemExit(f"empty submission: {path}")
        if [c.strip() for c in header] != ["ID", "Predicted"]:
            raise SystemExit(
                f"invalid header {header!r}; expected exactly ['ID', 'Predicted']. "
                "The 5-column sample_submission format is rejected by Kaggle."
            )
        n = 0
        seen = set()
        for row in reader:
            if not row:
                continue
            if len(row) != 2:
                raise SystemExit(f"row {n + 2}: expected 2 fields, got {len(row)}")
            ident = row[0].strip()
            if ident in seen:
                raise SystemExit(f"duplicate ID at row {n + 2}: {ident}")
            seen.add(ident)
            n += 1
    if n == 0:
        raise SystemExit("submission has no data rows")
    print(f"validated {path.name}: {n} rows, 2-column ID,Predicted")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", default=str(DEFAULT_SUBMISSION),
                    help="CSV path (absolute or relative to the repository root)")
    ap.add_argument("--slug", default=DEFAULT_SLUG, help="Kaggle competition slug")
    ap.add_argument("--message", default=DEFAULT_MESSAGE, help="submission message")
    ap.add_argument("--dry-run", action="store_true",
                    help="only validate the CSV; do not contact Kaggle")
    args = ap.parse_args()

    path = Path(args.submission)
    if not path.is_absolute():
        path = REPO / path
    validate_submission(path)

    if args.dry_run:
        print("dry run: no submission sent")
        return

    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    if not username or not key:
        raise SystemExit(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set. In this repository "
            "they are GitHub Actions secrets; see README.md."
        )
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise SystemExit("kaggle CLI not found; install with `pip install kaggle`")

    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key
    cmd = [kaggle, "competitions", "submit", "-c", args.slug,
           "-f", str(path), "-m", args.message]
    print("submitting:", args.slug, path.name)
    subprocess.run(cmd, check=True, env=env)

    subprocess.run([kaggle, "competitions", "submissions", "-c", args.slug],
                   check=False, env=env)


if __name__ == "__main__":
    main()
