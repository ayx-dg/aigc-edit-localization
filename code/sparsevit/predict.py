"""Inference for the final SparseViT run: Kaggle RLE submission + error stats.

Uses the *best* checkpoint and the threshold tuned on validation during
training. Produces, under --out-dir:
    submission.csv                  ID,Predicted        (Kaggle 2-col format)
    submission_sample_format.csv    ID,annotation,height,width,Usage
    confusion_matrix.json           val TP/FP/FN/TN at the tuned threshold
    val_per_image.csv               per-image val confusion + metrics
    val_probs_256.npz               compressed val probability maps (256px)
    predict_info.json               seed / threshold / timing / config
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
# Keep the official SparseViT directory ahead of ROOT: ROOT/SparseViT is a
# package, so if ROOT wins, `from SparseViT import SparseViT` binds a module
# instead of the class and SparseViT_Mul fails with "module is not callable".
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "SparseViT"))

from dataset import AIGCResizedDataset  # noqa: E402
from model import build_model            # noqa: E402


def rle_encode(mask: np.ndarray) -> str:
    """Column-major (Fortran), 1-based (start, length) RLE."""
    binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    flat = binary.reshape(-1, order="F")
    if not flat.any():
        return ""
    padded = np.concatenate(([0], flat, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0] + 1        # 1-based
    ends = np.where(diff == -1)[0]
    lengths = ends - starts + 1
    out = np.empty(2 * len(starts), dtype=int)
    out[0::2], out[1::2] = starts, lengths
    return " ".join(map(str, out.tolist()))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threshold", type=float, default=None,
                   help="if omitted, read best_info.json from --out-dir")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


@torch.no_grad()
def predict_probs(model, ds, batch_size, workers, device, use_amp):
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                     num_workers=workers, pin_memory=True)
    model.eval()
    out = []
    for xb, _ in dl:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xb)
        out.append(torch.sigmoid(logits.float()).cpu().numpy()[:, 0])
    return np.concatenate(out, axis=0)


def main():
    args = parse_args()
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    ck = torch.load(args.out_dir / "best.pt", map_location="cpu",
                    weights_only=False)
    train_args = ck.get("args", {})
    seed = int(train_args.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    thr = args.threshold
    info_path = args.out_dir / "best_info.json"
    if thr is None and info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        thr = float(info["metrics"]["threshold"])
    if thr is None:
        thr = float(ck.get("threshold", 0.5))
    print(f"[predict] threshold={thr:.3f} seed={seed} device={device} "
          f"amp={use_amp} best_epoch={ck.get('epoch', '?')}", flush=True)

    model = build_model(
        train_args.get("encoder_weights", ""), "sparsevit",
        img_size=int(train_args.get("size", 512)),
        drop_path=float(train_args.get("drop_path", 0.2)),
        s_blocks3=train_args.get("s_blocks3"),
        s_blocks4=train_args.get("s_blocks4"),
    ).to(device)
    model.load_state_dict(ck["model"])
    del ck

    meta_path = args.cache_dir / "meta.csv"
    img_dir = args.cache_dir / "imgs"
    mask_dir = args.cache_dir / "masks"
    t0 = time.time()

    # ---------- validation: confusion matrix + per-image stats ----------
    val_ds = AIGCResizedDataset(meta_path, img_dir, mask_dir, split="val",
                                size=args.size)
    val_probs = predict_probs(model, val_ds, args.batch_size, args.workers,
                              device, use_amp)
    tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_rows = []
    for i in range(len(val_ds)):
        _, y = val_ds[i]
        gt = (y[0].numpy() > 0.5)
        pr = (val_probs[i] > thr)
        tp = int((gt & pr).sum()); fp = int((~gt & pr).sum())
        fn = int((gt & ~pr).sum()); tn = int((~gt & ~pr).sum())
        for k, v in (("tp", tp), ("fp", fp), ("fn", fn), ("tn", tn)):
            tot[k] += v
        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        per_rows.append({
            "ID": int(val_ds.df.iloc[i]["ID"]),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "gt_pos": int(gt.sum()), "pred_pos": int(pr.sum()),
            "iou": tp / denom_iou if denom_iou else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "dice": 2 * tp / denom_dice if denom_dice else 1.0,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
        })
    with (args.out_dir / "val_per_image.csv").open("w", newline="",
                                                   encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0]))
        w.writeheader(); w.writerows(per_rows)
    prec = tot["tp"] / max(1, tot["tp"] + tot["fp"])
    spec = tot["tn"] / max(1, tot["tn"] + tot["fp"])
    rec = tot["tp"] / max(1, tot["tp"] + tot["fn"])
    cm = {
        "threshold": thr,
        "counts": tot,
        "precision": prec, "recall": rec, "specificity": spec,
        "false_positive_rate": 1.0 - spec,
        "normalized": [
            [tot["tn"] / max(1, tot["tn"] + tot["fp"]),
             tot["fp"] / max(1, tot["tn"] + tot["fp"])],
            [tot["fn"] / max(1, tot["fn"] + tot["tp"]),
             tot["tp"] / max(1, tot["fn"] + tot["tp"])],
        ],
        "row_true": [0, 1], "col_pred": [0, 1],
    }
    (args.out_dir / "confusion_matrix.json").write_text(
        json.dumps(cm, indent=2), encoding="utf-8")
    print(f"[confusion] TP={tot['tp']} FP={tot['fp']} FN={tot['fn']} "
          f"TN={tot['tn']} precision={prec:.4f} recall={rec:.4f} "
          f"specificity={spec:.4f}", flush=True)

    # Small compressed probability cache for extra error analysis.
    small = np.stack([cv2_resize(p, 256) for p in val_probs])
    np.savez_compressed(args.out_dir / "val_probs_256.npz", probs=small)

    # ---------- test: RLE submission ----------
    import pandas as pd
    meta = pd.read_csv(meta_path)
    test = meta[meta["split"] == "test"].reset_index(drop=True)
    from torch.utils.data import Dataset

    class TestDS(Dataset):
        def __init__(self, df):
            self.df = df
            self.mean = torch.tensor(AIGCResizedDataset.MEAN).view(3, 1, 1)
            self.std = torch.tensor(AIGCResizedDataset.STD).view(3, 1, 1)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            row = self.df.iloc[i]
            arr = np.load(img_dir / f"{int(row.ID):06d}.npy").astype(np.float32) / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            x = F.interpolate(x, size=(args.size, args.size), mode="bilinear",
                              align_corners=False).squeeze(0)
            # predict_probs expects (image, target) batches; test has no masks.
            return (x - self.mean) / self.std, torch.zeros(1)

    test_probs = predict_probs(model, TestDS(test), args.batch_size,
                               args.workers, device, use_amp)
    rows_2col, rows_5col = [], []
    empty = 0
    for i, row in test.iterrows():
        h, w = int(row.orig_h), int(row.orig_w)
        p = torch.from_numpy(test_probs[i]).unsqueeze(0).unsqueeze(0)
        p = F.interpolate(p, size=(h, w), mode="bilinear",
                          align_corners=False)[0, 0].numpy()
        rle = rle_encode((p > thr).astype(np.uint8))
        if not rle:
            empty += 1
        ident6 = f"{int(row.ID):06d}"
        rows_2col.append({"ID": ident6, "Predicted": rle})
        rows_5col.append({"ID": ident6, "annotation": rle,
                          "height": h, "width": w, "Usage": "Public"})
    with (args.out_dir / "submission.csv").open("w", newline="",
                                                encoding="utf-8") as f:
        w = csv.DictWriter(f, ["ID", "Predicted"]); w.writeheader()
        w.writerows(rows_2col)
    with (args.out_dir / "submission_sample_format.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["ID", "annotation", "height", "width", "Usage"])
        w.writeheader(); w.writerows(rows_5col)
    print(f"[submission] {len(rows_2col)} rows, {empty} empty masks, "
          f"threshold={thr:.3f}", flush=True)

    best_epoch = (json.loads(info_path.read_text(encoding="utf-8")).get("epoch")
                  if info_path.exists() else None)
    (args.out_dir / "predict_info.json").write_text(json.dumps({
        "seed": seed,
        "threshold": thr,
        "best_epoch": best_epoch,
        "test_images": len(test),
        "empty_masks": empty,
        "confusion_matrix": cm,
        "inference_time_sec": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")


def cv2_resize(p, size):
    """Cheap bilinear resize without cv2/PIL dependency."""
    t = torch.from_numpy(p).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(size, size), mode="bilinear",
                         align_corners=False)[0, 0].numpy().astype(np.float16)


if __name__ == "__main__":
    main()
