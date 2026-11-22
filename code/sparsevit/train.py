"""Full training for the minimal U-Net + ResNet34 (ImageNet) model.

Optimises / reports the Kaggle metric from metrics.py:
    Score = 0.7 * mIoU + 0.3 * mRecall   (per image, then averaged)

Examples
--------
# full run on the GPU box (detached launcher handles long runs)
python train.py --cache-dir C:/path/cache --out-dir C:/path/out --epochs 40

# quick smoke (same code path, capped steps)
python train.py --cache-dir C:/path/cache --out-dir C:/path/out \
    --steps 10 --size 256 --batch-size 4 --limit-train 64 --limit-val 16
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import AIGCResizedDataset
from dataset_patch import PatchDataset
from metrics import evaluate, sweep_threshold
from model import build_model

HISTORY_FIELDS = [
    "epoch", "train_loss", "val_loss", "val_dice", "miou", "mrecall",
    "score", "threshold", "lr", "elapsed_sec", "is_best",
]


class BCEDiceLoss(nn.Module):
    """Raw-logit BCE + soft Dice (matches the original project's bce_dice)."""

    def __init__(self, dice_w: float = 1.0, eps: float = 1.0):
        super().__init__()
        self.dice_w, self.eps = dice_w, eps

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        inter = (probs * target).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)
        return bce + self.dice_w * dice.mean()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=here / "cache")
    p.add_argument("--patch-dir", type=Path, default=None,
                   help="native 512px patch cache from make_patches.py; "
                        "when set, replaces the downscaled whole-image cache")
    p.add_argument("--out-dir", type=Path, default=here / "out")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps", type=int, default=None,
                   help="cap total optimizer steps (for smoke runs)")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="linear LR warmup steps before cosine decay (0=off)")
    p.add_argument("--drop-path", type=float, default=0.2,
                   help="SparseViT stochastic-depth rate (0.2=official default)")
    p.add_argument("--s-blocks3", default=None,
                   help="SparseViT stage-3 sparse blocks, e.g. 4,2,1,1 "
                        "(official default 8,4,2,1)")
    p.add_argument("--s-blocks4", default=None,
                   help="SparseViT stage-4 sparse blocks, e.g. 1,1 "
                        "(official default 2,1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--loss", choices=("bce", "bce_dice"), default="bce_dice")
    p.add_argument("--best-metric", choices=("score", "dice", "miou", "mrecall"),
                   default="score", help="metric used to pick best.pt")
    p.add_argument("--save-every-steps", type=int, default=200,
                   help="also checkpoint every N steps (0 disables)")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--arch", default="unet",
                   choices=("unet", "unetpp", "segformer", "deeplabv3plus",
                            "effnet_unet", "sparsevit"))
    p.add_argument("--cudnn", action="store_true",
                   help="keep cuDNN enabled (off by default: the ac1 torch "
                        "wheel has a broken cudnnGetVersion symbol)")
    p.add_argument("--no-amp", action="store_true", help="disable fp16 autocast")
    return p.parse_args()


@torch.no_grad()
def validate(model, val_dl, device, criterion, use_amp=False, sweep_cap=128):
    model.eval()
    probs, masks = [], []
    loss_sum, n_seen = 0.0, 0
    for xb, yb in val_dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)
        loss_sum += float(loss.item()) * xb.size(0)
        n_seen += xb.size(0)
        p = torch.sigmoid(logits).float().cpu().numpy()
        for i in range(p.shape[0]):
            probs.append(p[i, 0].astype(np.float16))
            masks.append(yb[i, 0].cpu().numpy().astype(np.uint8))

    if len(probs) > sweep_cap:
        idx = np.linspace(0, len(probs) - 1, sweep_cap).astype(int)
        sub_p = [probs[i] for i in idx]
        sub_m = [masks[i] for i in idx]
    else:
        sub_p, sub_m = probs, masks
    best_t, _ = sweep_threshold(sub_p, sub_m)
    result = evaluate(probs, masks, best_t)
    result["val_loss"] = loss_sum / max(1, n_seen)
    result["threshold"] = best_t
    result["gt_pos"] = float(np.mean([m.mean() for m in masks]))
    result["pred_pos"] = float(np.mean([(p > best_t).mean() for p in probs]))
    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    if not args.cudnn:
        # ac1's torch 2.10 + CUDA 13 wheel fails to load cudnnGetVersion;
        # plain CUDA convolutions work fine, so skip cuDNN by default.
        torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"[env] torch={torch.__version__} cuda_build={torch.version.cuda} "
          f"device={device} amp={use_amp} cudnn={torch.backends.cudnn.enabled}",
          flush=True)
    if device.type == "cuda":
        print(f"[env] gpu={torch.cuda.get_device_name(0)}", flush=True)

    if args.patch_dir is not None:
        print(f"[data] native patch cache: {args.patch_dir}", flush=True)
        train_ds = PatchDataset(args.patch_dir, "train", size=args.size,
                                augment=True, limit=args.limit_train,
                                seed=args.seed)
        val_ds = PatchDataset(args.patch_dir, "val", size=args.size,
                              augment=False, limit=args.limit_val,
                              seed=args.seed)
    else:
        cache = args.cache_dir
        meta, imgs, masks = cache / "meta.csv", cache / "imgs", cache / "masks"
        for p in (meta, imgs, masks):
            if not p.exists():
                raise FileNotFoundError(f"missing cache path: {p}")
        train_ds = AIGCResizedDataset(meta, imgs, masks, split="train",
                                      size=args.size, limit=args.limit_train,
                                      seed=args.seed)
        val_ds = AIGCResizedDataset(meta, imgs, masks, split="val",
                                    size=args.size, limit=args.limit_val,
                                    seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True,
                          pin_memory=(device.type == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers,
                        pin_memory=(device.type == "cuda"))
    steps_per_epoch = len(train_dl)
    total_steps = args.steps if args.steps else args.epochs * steps_per_epoch
    print(f"[data] train={len(train_ds)} val={len(val_ds)} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps} "
          f"size={args.size} batch={args.batch_size}", flush=True)

    s3 = [int(v) for v in args.s_blocks3.split(",")] if args.s_blocks3 else None
    s4 = [int(v) for v in args.s_blocks4.split(",")] if args.s_blocks4 else None
    if args.arch == "sparsevit":
        print(f"[params] img_size={args.size} drop_path={args.drop_path} "
              f"s_blocks3={s3} s_blocks4={s4}", flush=True)
    model = build_model(args.encoder_weights, args.arch,
                        img_size=args.size, drop_path=args.drop_path,
                        s_blocks3=s3, s_blocks4=s4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = BCEDiceLoss() if args.loss == "bce_dice" else nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    if args.warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, end_factor=1.0,
            total_iters=args.warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, total_steps - args.warmup_steps))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [warmup, cosine], milestones=[args.warmup_steps])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "args": vars(args),
        "resolved": {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "amp": use_amp,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "s_blocks3": s3,
            "s_blocks4": s4,
            "seed": args.seed,
        },
    }
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False, default=str)
    last_path = args.out_dir / "last.pt"
    best_path = args.out_dir / "best.pt"

    start_epoch, global_step, best_score, best_threshold = 0, 0, -1.0, 0.5
    if args.resume and args.resume.exists():
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch = int(ck["epoch"]) + 1
        global_step = int(ck.get("step", 0))
        best_score = float(ck.get("best_score", -1.0))
        best_threshold = float(ck.get("threshold", 0.5))
        print(f"[resume] {args.resume} epoch={start_epoch} "
              f"step={global_step} best_score={best_score:.4f}", flush=True)

    # Per-epoch history CSV. On resume, drop rows from epochs we will redo.
    history_path = args.out_dir / "history.csv"
    history = []
    if args.resume and history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as f:
            history = list(csv.DictReader(f))
        history = [r for r in history if int(float(r["epoch"])) < start_epoch]
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(history)
    header_written = history_path.exists() and history_path.stat().st_size > 0

    def log_history(row):
        nonlocal header_written
        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
            if not header_written:
                writer.writeheader()
                header_written = True
            writer.writerow(row)
            f.flush()

    def save(path, **extra):
        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "best_score": best_score,
            "threshold": best_threshold,
            "args": vars(args),
            **extra,
        }, path)
        print(f"  -> checkpoint {path}", flush=True)

    print(f"[model] params={n_params} loss={args.loss} lr={args.lr}", flush=True)
    t0 = time.time()
    stop = False
    try:
        for epoch in range(start_epoch, args.epochs):
            if stop:
                break
            model.train()
            run_loss = 0.0
            n_batch = 0
            for xb, yb in train_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()

                global_step += 1
                run_loss += loss.item()
                n_batch += 1
                if global_step == 1 or global_step % 100 == 0:
                    print(f"[train] ep={epoch+1}/{args.epochs} "
                          f"step={global_step}/{total_steps} loss={loss.item():.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} "
                          f"elapsed={time.time()-t0:.0f}s", flush=True)
                if args.save_every_steps and global_step % args.save_every_steps == 0:
                    save(last_path, loss=loss.item())
                if args.steps and global_step >= args.steps:
                    stop = True
                    break

            if stop and not args.steps:
                break

            # ---- validation: loss + Dice + IoU + Recall + Score ----
            res = validate(model, val_dl, device, criterion, use_amp)
            mean_loss = run_loss / max(1, n_batch)
            is_best = res[args.best_metric] > best_score
            print(f"[val] ep={epoch+1}/{args.epochs} "
                  f"train_loss={mean_loss:.4f} val_loss={res['val_loss']:.4f} "
                  f"dice={res['dice']:.4f} miou={res['miou']:.4f} "
                  f"mrecall={res['mrecall']:.4f} score={res['score']:.4f} "
                  f"thr={res['threshold']:.2f} "
                  f"gt_pos={res['gt_pos']:.3f} pred_pos={res['pred_pos']:.3f} "
                  f"elapsed={time.time()-t0:.0f}s"
                  f"{'  -> BEST' if is_best else ''}", flush=True)
            log_history({
                "epoch": epoch + 1,
                "train_loss": f"{mean_loss:.6f}",
                "val_loss": f"{res['val_loss']:.6f}",
                "val_dice": f"{res['dice']:.6f}",
                "miou": f"{res['miou']:.6f}",
                "mrecall": f"{res['mrecall']:.6f}",
                "score": f"{res['score']:.6f}",
                "threshold": f"{res['threshold']:.3f}",
                "lr": f"{sched.get_last_lr()[0]:.3e}",
                "elapsed_sec": f"{time.time()-t0:.1f}",
                "is_best": int(is_best),
            })
            save(last_path, loss=mean_loss, val=res)
            if is_best:
                best_score = res[args.best_metric]
                best_threshold = res["threshold"]
                save(best_path, loss=mean_loss, val=res)
                info = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "best_metric": args.best_metric,
                    "best_value": best_score,
                    "seed": args.seed,
                    "train_samples": len(train_ds),
                    "val_samples": len(val_ds),
                    "training_time_sec": round(time.time() - t0, 1),
                    "train_loss": mean_loss,
                    "val_loss": res["val_loss"],
                    "metrics": res,
                }
                with (args.out_dir / "best_info.json").open(
                        "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=2, ensure_ascii=False)
                print(f"  -> new best {args.best_metric}={best_score:.4f} "
                      f"thr={best_threshold:.2f}", flush=True)
    except KeyboardInterrupt:
        print("[interrupt] saving emergency checkpoint...", flush=True)
        save(last_path)
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", flush=True)
        save(last_path)
        raise

    total_sec = time.time() - t0
    with (args.out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "steps": global_step,
            "best_metric": args.best_metric,
            "best_value": best_score,
            "best_threshold": best_threshold,
            "training_time_sec": round(total_sec, 1),
            "config": run_config,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"[done] steps={global_step} best_score={best_score:.4f} "
          f"best_thr={best_threshold:.2f} total={total_sec:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
