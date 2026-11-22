"""Dataset for the native-resolution 512x512 patch cache (PNG / uint8).

Patches are produced by make_patches.py: lossless WebP crop -> PNG image,
lossless RLE crop -> PNG mask. No resize is applied here, so the model sees
true native pixels (this is the whole point of the patch experiment).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class PatchDataset(Dataset):
    def __init__(self, patch_dir, split, size=512, augment=False, limit=None,
                 seed=42, lossless_check=True):
        self.root = Path(patch_dir)
        self.df = pd.read_csv(self.root / f"manifest_{split}.csv",
                              dtype={"ID": str})
        self.df = self.df[self.df["status"] == "OK"].reset_index(drop=True)
        if limit is not None and limit < len(self.df):
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(self.df), size=limit, replace=False))
            self.df = self.df.iloc[idx].reset_index(drop=True)
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.size = int(size)
        self.augment = augment
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)
        self.lossless_check = lossless_check
        # Sanity: every patch must be exactly size x size.
        if lossless_check and len(self.df):
            a = np.asarray(Image.open(self.img_dir / f"{self.df.iloc[0]['ID']}_0.png"))
            assert a.shape[:2] == (self.size, self.size), (
                f"patch size {a.shape[:2]} != {self.size}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        # Manifest column 'k' stores the patch index used as filename suffix.
        stem = f"{int(row['ID']):06d}_{int(row['k'])}"
        img = np.asarray(Image.open(self.img_dir / f"{stem}.png").convert("RGB"),
                         dtype=np.uint8)
        mask = np.asarray(Image.open(self.mask_dir / f"{stem}.png"),
                          dtype=np.uint8) > 127

        if self.augment:
            # Dihedral group D4: 4 rotations x 2 flips. Lossless, mask-safe.
            k = int(np.random.randint(0, 4))
            if k:
                img = np.rot90(img, k)
                mask = np.rot90(mask, k)
            if np.random.rand() < 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)

        x = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(np.array(mask, copy=True)).float().unsqueeze(0)
        x = (x - self.mean) / self.std
        return x, y
