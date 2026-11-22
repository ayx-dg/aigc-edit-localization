import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd


class AIGCDataset(Dataset):
    def __init__(self, meta_csv, image_dir, mask_dir):
        self.df = pd.read_csv(meta_csv)
        if "filename" not in self.df.columns and "ID" in self.df.columns:
            # cache/meta.csv 只有 ID，缓存文件名为 000001.npy 这种格式
            self.df = self.df.copy()
            self.df["filename"] = self.df["ID"].map(lambda i: f"{int(i):06d}.npy")
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        name = row["filename"]

        image = np.load(self.image_dir / name)
        mask = np.load(self.mask_dir / name)

        # HWC -> CHW
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # HW -> 1HW
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask


class AIGCResizedDataset(Dataset):
    """Fixed-size, ImageNet-normalized train/val pairs straight from cache.

    No albumentations dependency: resizing is done with torch interpolate, so
    the same code runs on the local CPU box and on the Windows GPU box.
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, meta_csv, image_dir, mask_dir, split=None, size=256,
                 limit=None, seed=42):
        self.df = pd.read_csv(meta_csv)
        if "filename" not in self.df.columns and "ID" in self.df.columns:
            self.df = self.df.copy()
            self.df["filename"] = self.df["ID"].map(lambda i: f"{int(i):06d}.npy")
        if split is not None:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if limit is not None and limit < len(self.df):
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(self.df), size=limit, replace=False))
            self.df = self.df.iloc[idx].reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.size = int(size)
        self.mean = torch.tensor(self.MEAN).view(3, 1, 1)
        self.std = torch.tensor(self.STD).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df.iloc[idx]["filename"]
        image = np.load(self.image_dir / name).astype(np.float32) / 255.0
        mask = np.load(self.mask_dir / name).astype(np.float32)

        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        y = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear",
                          align_corners=False).squeeze(0)
        y = F.interpolate(y, size=(self.size, self.size), mode="nearest").squeeze(0)
        x = (x - self.mean) / self.std
        return x, y


def main():
    import argparse
    from torch.utils.data import DataLoader, Subset

    default_cache = Path(__file__).resolve().parent / "cache"
    parser = argparse.ArgumentParser(description="检查缓存数据集")
    parser.add_argument("--cache-dir", type=Path, default=default_cache)
    parser.add_argument("--samples", type=int, default=3,
                        help="实际加载的样本数，0 表示只检查文件是否存在")
    args = parser.parse_args()

    meta_csv = args.cache_dir / "meta.csv"
    image_dir = args.cache_dir / "imgs"
    mask_dir = args.cache_dir / "masks"

    ds = AIGCDataset(meta_csv, image_dir, mask_dir)
    print(f"meta: {meta_csv}, 记录数: {len(ds)}")
    if "split" in ds.df.columns:
        print("split 分布:", ds.df["split"].value_counts().to_dict())

    # 先只检查文件是否齐全，不加载数据
    names = ds.df["filename"].unique()
    missing_imgs = [n for n in names if not (image_dir / n).is_file()]
    missing_masks = [n for n in names if not (mask_dir / n).is_file()]
    print(f"缺失图片: {len(missing_imgs)} {missing_imgs[:5]}")
    print(f"缺失 mask: {len(missing_masks)} {missing_masks[:5]}")

    # 只在文件齐全的样本上做加载检查（test split 没有 mask）
    bad_names = set(missing_imgs) | set(missing_masks)
    valid_idx = [i for i, n in enumerate(ds.df["filename"]) if n not in bad_names]
    print(f"可用样本: {len(valid_idx)}/{len(ds)}")
    if args.samples <= 0:
        return

    # 通过 Dataset / DataLoader 实际加载少量样本，检查形状与数值范围
    loader = DataLoader(Subset(ds, valid_idx), batch_size=1, shuffle=False, num_workers=0)
    for i, (image, mask) in enumerate(loader):
        if i >= args.samples:
            break
        image, mask = image[0], mask[0]
        row = ds.df.iloc[valid_idx[i]]
        expected = (int(row["cache_h"]), int(row["cache_w"]))
        ok = tuple(mask.shape[-2:]) == expected
        print(
            f"[{i}] {row['filename']} split={row.get('split', '-')} "
            f"image={tuple(image.shape)} [{image.min():.3f}, {image.max():.3f}] "
            f"mask={tuple(mask.shape)} "
            f"unique={torch.unique(mask).tolist()} pos={mask.mean().item():.4f} "
            f"size={tuple(mask.shape[-2:])}/{expected} {'OK' if ok else 'MISMATCH'}"
        )


if __name__ == "__main__":
    main()
