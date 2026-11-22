"""Kaggle scoring for AIGC Detection (computed per image, then averaged).

Score = 0.7 * mIoU + 0.3 * mRecall
Empty-vs-empty -> 1 for both metrics; exactly one empty -> 0 for both.
"""
from __future__ import annotations

import numpy as np


def image_iou_recall(y_true: np.ndarray, y_prob: np.ndarray,
                     threshold: float = 0.5) -> tuple[float, float]:
    t = np.asarray(y_true) > 0.5
    p = np.asarray(y_prob) > threshold
    n_t, n_p = int(t.sum()), int(p.sum())
    if n_t == 0 and n_p == 0:
        return 1.0, 1.0
    if n_t == 0 or n_p == 0:
        return 0.0, 0.0
    inter = float((t & p).sum())
    iou = inter / float((t | p).sum())
    recall = inter / float(n_t)
    return float(iou), float(recall)


def image_dice(y_true: np.ndarray, y_prob: np.ndarray,
               threshold: float = 0.5) -> float:
    """Per-image Dice, using the same empty-mask convention as IoU/Recall."""
    t = np.asarray(y_true) > 0.5
    p = np.asarray(y_prob) > threshold
    n_t, n_p = int(t.sum()), int(p.sum())
    if n_t == 0 and n_p == 0:
        return 1.0
    if n_t == 0 or n_p == 0:
        return 0.0
    inter = float((t & p).sum())
    return float(2.0 * inter / (n_t + n_p))


def evaluate(probs, masks, threshold: float = 0.5) -> dict:
    """Return mean Dice / IoU / Recall and the combined competition score."""
    dices, ious, recalls = [], [], []
    for p, m in zip(probs, masks):
        i, r = image_iou_recall(m, p, threshold)
        dices.append(image_dice(m, p, threshold))
        ious.append(i)
        recalls.append(r)
    mdice, miou, mrec = (float(np.mean(x)) for x in (dices, ious, recalls))
    return {"dice": mdice, "miou": miou, "mrecall": mrec,
            "score": 0.7 * miou + 0.3 * mrec}


def sweep_threshold(probs, masks, lo: float = 0.1, hi: float = 0.91,
                    step: float = 0.05) -> tuple[float, dict]:
    """Pick the threshold maximising the competition score on validation."""
    best_t, best = lo, {"score": -1.0}
    for t in np.arange(lo, hi, step):
        res = evaluate(probs, masks, float(t))
        if res["score"] > best["score"]:
            best_t, best = float(t), res
    return best_t, best
