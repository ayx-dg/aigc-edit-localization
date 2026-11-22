#!/usr/bin/env python3
"""Gate a proposed CSV against our historical best public Score.

The baseline is queried at runtime from **our own** Kaggle submission history
(not the global leaderboard), with a local floor as offline fallback. A
candidate replaces the canonical competition CSV only when its public Score is
strictly greater than that baseline. Otherwise the canonical CSV and its score
sidecar are restored from the archived historical best.

Outputs (stdout, and to ``--github-output`` when provided)::

    kaggle_best=0.92213
    candidate_score=0.80870
    decision=KEEP
    restore_file=submissions/historical_best_0.92213.csv

Usage::

    python scripts/gate_best_submission.py
    python scripts/gate_best_submission.py --score 0.99 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SUBMISSION = REPO / "submissions" / "best_submission.csv"
DEFAULT_SCORE_FILE = REPO / "submissions" / "best_submission.score.json"
DEFAULT_REGISTRY = REPO / "submissions" / "score_registry.json"
DEFAULT_SLUG = "ai-edited-image-localization-cs-5489"
DEFAULT_FLOOR = 0.92213  # historical best, ref 58415875


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def query_kaggle_best(slug: str, kaggle_bin: str | None) -> tuple[float | None, str]:
    """Return the best public Score among our COMPLETE Kaggle submissions."""
    kaggle = kaggle_bin or shutil.which("kaggle")
    if kaggle is None:
        return None, "kaggle CLI not found"
    env = os.environ.copy()
    cmd = [kaggle, "competitions", "submissions", "-c", slug,
           "--csv", "--page-size", "200"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=180, env=env)
    except Exception as exc:  # network / timeout
        return None, f"kaggle query failed: {exc}"
    if proc.returncode != 0:
        return None, f"kaggle query returned {proc.returncode}: {proc.stderr.strip()[:200]}"
    try:
        rows = list(csv.DictReader(proc.stdout.splitlines()))
    except csv.Error as exc:
        return None, f"cannot parse kaggle CSV: {exc}"
    best = None
    for row in rows:
        if "COMPLETE" not in (row.get("status") or "").upper():
            continue
        try:
            score = float(row.get("publicScore") or "")
        except ValueError:
            continue
        if best is None or score > best:
            best = score
    if best is None:
        return None, "no completed submission with a public Score"
    return best, ""


def load_archive(registry: Path) -> list[tuple[float, str]]:
    """Return (score, relative file) pairs from the score registry."""
    try:
        data = json.loads(registry.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for entry in data.get("archive", []):
        try:
            out.append((float(entry["score"]), str(entry["file"])))
        except (KeyError, ValueError, TypeError):
            continue
    return sorted(out, reverse=True)


def pick_restore(archive: list[tuple[float, str]], best: float) -> str | None:
    """Archived CSV with the highest score not exceeding the Kaggle best."""
    eligible = [f for s, f in archive if s <= best + 1e-12]
    return eligible[0] if eligible else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", default=str(DEFAULT_SUBMISSION))
    ap.add_argument("--score-file", default=str(DEFAULT_SCORE_FILE))
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--slug", default=DEFAULT_SLUG)
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    ap.add_argument("--score", type=float, default=None,
                    help="override candidate score (workflow_dispatch / tests)")
    ap.add_argument("--kaggle", default=None,
                    help="explicit kaggle CLI path (tests)")
    ap.add_argument("--kaggle-best", type=float, default=None,
                    help="override the queried baseline (offline tests only)")
    ap.add_argument("--github-output", default=None,
                    help="append key=value lines to this GitHub Actions output file")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the decision but do not modify any file")
    args = ap.parse_args()

    submission = resolve(args.submission)
    score_file = resolve(args.score_file)
    registry = resolve(args.registry)

    sidecar = {}
    if score_file.is_file():
        try:
            sidecar = json.loads(score_file.read_text(encoding="utf-8"))
        except Exception:
            sidecar = {}

    if args.score is not None:
        candidate = args.score  # explicit simulation / dispatch override
    else:
        try:
            candidate = float(sidecar["score"])
        except (KeyError, ValueError, TypeError) as exc:
            raise SystemExit(
                f"candidate score missing: {score_file} ({exc}). "
                "Commit a sidecar score JSON or pass --score."
            )
        expected = sidecar.get("sha256")
        if expected:
            actual = hashlib.sha256(submission.read_bytes()).hexdigest()
            if actual != str(expected):
                raise SystemExit(
                    f"score sidecar checksum mismatch for {submission.name}: "
                    "the declared Score does not describe this CSV. "
                    "Update the CSV and sidecar together."
                )

    if args.kaggle_best is not None:
        queried, note = args.kaggle_best, "override"
    else:
        queried, note = query_kaggle_best(args.slug, args.kaggle)
    if queried is None:
        print(f"warning: {note}; using floor {args.floor}", file=sys.stderr)
        queried = args.floor
    kaggle_best = max(args.floor, queried)

    submit = candidate > kaggle_best
    decision = "SUBMIT" if submit else "KEEP"
    restore_file = ""
    if not submit:
        rel = pick_restore(load_archive(registry), kaggle_best)
        if rel is None:
            raise SystemExit(
                f"no archived CSV in {registry.name} matches baseline {kaggle_best}"
            )
        restore_file = str(Path("submissions") / Path(rel).name)

    outputs = {
        "kaggle_best": f"{kaggle_best:.5f}",
        "candidate_score": f"{candidate:.5f}",
        "decision": decision,
        "restore_file": restore_file,
    }
    if args.github_output:
        with Path(args.github_output).open("a", encoding="utf-8") as fh:
            for key, value in outputs.items():
                fh.write(f"{key}={value}\n")
    for key, value in outputs.items():
        print(f"{key}={value}")

    if args.dry_run:
        print("dry run: no repository file was modified")
        return

    if submit:
        # Archive the newly accepted CSV so a future run can restore it.
        archive = REPO / "submissions" / f"best_{candidate:.5f}.csv"
        shutil.copy2(submission, archive)
        data = {"best_score": candidate, "archive": []}
        if registry.is_file():
            try:
                data = json.loads(registry.read_text(encoding="utf-8"))
            except Exception:
                pass
        entries = data.setdefault("archive", [])
        if not any(abs(float(e.get("score", 0)) - candidate) < 1e-12
                   for e in entries if isinstance(e, dict)):
            entries.append({"score": candidate, "file": archive.name})
        data["best_score"] = candidate
        registry.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"archived accepted CSV: {archive.relative_to(REPO)}")
    else:
        shutil.copy2(REPO / restore_file, submission)
        digest = hashlib.sha256(submission.read_bytes()).hexdigest()
        score_file.write_text(json.dumps({
            "score": kaggle_best,
            "source": "kaggle_submission_history",
            "sha256": digest,
            "note": "Score of the CSV currently in best_submission.csv.",
        }, indent=2) + "\n", encoding="utf-8")
        print(f"restored canonical CSV from {restore_file}; "
              f"score sidecar set to {kaggle_best:.5f}")


if __name__ == "__main__":
    main()
