#!/usr/bin/env python3
"""Combine global and native probability maps into a weighted ensemble NPZ.

The saved maps are uint8 in [0, 255], so averaging is done after division by
255 and the result is written back in the same uint8 format.

Usage:
  python scripts/make_ensemble_probs.py GLOBAL.npz NATIVE.npz OUT.npz \
      --w-global 0.4 --w-native 0.6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("global_probs", type=Path)
    ap.add_argument("native_probs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--w-global", type=float, default=0.4)
    ap.add_argument("--w-native", type=float, default=0.6)
    args = ap.parse_args()

    g = np.load(args.global_probs)
    p = np.load(args.native_probs)
    ids_g = [str(s).strip() for s in g["ids"]]
    ids_p = [str(s).strip() for s in p["ids"]]
    if ids_g != ids_p:
        raise SystemExit("global and native NPZ files contain different ID lists")

    probs = (args.w_global * g["probs"].astype(np.float32) / 255.0
             + args.w_native * p["probs"].astype(np.float32) / 255.0)
    out = np.clip(np.rint(probs * 255.0), 0, 255).astype(np.uint8)
    np.savez_compressed(args.out, probs=out, ids=np.asarray(ids_g))
    print(f"wrote {args.out}: {len(ids_g)} maps, weights "
          f"{args.w_global}/{args.w_native}")


if __name__ == "__main__":
    main()
