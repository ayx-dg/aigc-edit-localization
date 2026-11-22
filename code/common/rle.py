"""Run-Length Encoding for the masks.

Convention (verified against the ground-truth data):
  * Binary mask is flattened in COLUMN-MAJOR (Fortran) order.
  * Positions are 1-BASED.
  * An RLE string is pairs of (start, length):  "3 4 10 2"  = runs starting at
    positions 3 and 10 with lengths 4 and 2.
  * Empty string => no edited pixels.

Reference round-trip property:  decode(encode(mask)) == mask.
"""
from __future__ import annotations

import numpy as np


def rle_decode(annotation: str, height: int, width: int) -> np.ndarray:
    """Decode an RLE annotation into a binary (H, W) uint8 mask.

    Column-major, 1-based, per config.RLE_ORDER / RLE_BASE.
    """
    mask_flat = np.zeros(height * width, dtype=np.uint8)
    if annotation is None:
        return mask_flat.reshape(height, width)

    s = str(annotation).strip()
    if s == "":
        return mask_flat.reshape(height, width)

    vals = list(map(int, s.split()))
    starts = vals[0::2]
    lengths = vals[1::2]
    for start, length in zip(starts, lengths):
        # 1-based -> 0-based inclusive slice
        mask_flat[start - 1: start - 1 + length] = 1

    return mask_flat.reshape((height, width), order="F")


def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary (H, W) mask into an RLE string (column-major, 1-based).

    Accepts float/uint arrays; anything > 0.5 is treated as foreground.
    """
    mask = np.asarray(mask)
    binary = (mask > 0.5).astype(np.uint8)
    # Column-major (Fortran) flatten: flat[k] = binary[k % H, k // H].
    flat = binary.reshape(-1, order="F")

    if flat.sum() == 0:
        return ""

    # Find run boundaries: positions where value changes.
    padded = np.concatenate(([0], flat, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0] + 1          # 1-based start
    ends = np.where(diffs == -1)[0]               # 1-based inclusive end
    lengths = ends - starts + 1

    runs = np.empty(2 * len(starts), dtype=int)
    runs[0::2] = starts
    runs[1::2] = lengths
    return " ".join(map(str, runs.tolist()))


def rle_encode_from_probs(probs: np.ndarray, height: int, width: int,
                          threshold: float = 0.5) -> str:
    """Encode a probability map that is at a DIFFERENT resolution than the
    target. Resizes to (height, width) first, then thresholds, then encodes.

    `probs` shape is (h_in, w_in) or (1, h_in, w_in).
    """
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim == 3:
        probs = probs[0]
    h_in, w_in = probs.shape
    if (h_in, w_in) != (height, width):
        # Bilinear resize (PIL is fine; avoid importing torch just for this).
        from PIL import Image
        img = Image.fromarray((probs * 255.0).astype(np.uint8))
        img = img.resize((width, height), Image.BILINEAR)
        probs = np.asarray(img, dtype=np.float32) / 255.0
    mask = (probs > threshold).astype(np.uint8)
    return rle_encode(mask)


def rle_area(annotation: str) -> int:
    """Total number of foreground pixels encoded in an RLE string."""
    if annotation is None or str(annotation).strip() == "":
        return 0
    vals = list(map(int, str(annotation).split()))
    return int(sum(vals[1::2]))


def roundtrip_test(df) -> bool:
    """Assert decode(encode(mask)) == mask for every annotation in df.

    df must have columns: annotation, height, width.
    Returns True if all pass; otherwise raises.
    """
    n = 0
    for _, row in df.iterrows():
        h, w = int(row["height"]), int(row["width"])
        mask = rle_decode(row["annotation"], h, w)
        back = rle_encode(mask)
        a = str(row["annotation"]).strip() if row["annotation"] is not None else ""
        b = back.strip()
        assert a == b, f"roundtrip mismatch on row {n} (h={h}, w={w})"
        n += 1
    return True
