# Historical native-resolution pipeline (~0.92 public score)

These files are the reusable core of the historical Kaggle submissions scoring
0.90638–0.92213. They were previously referred to as `aigc/` in the report;
the report's `aigc/native.py:81-108` reference corresponds to `native.py` here.

| File | Purpose |
|---|---|
| `native.py` | Native-pixel 512×512 crop sampling, edit-centred training crops, full-resolution tiled validation, official Score sweep |
| `core.py` | Tiny GroupNorm U-Net and BCE+Dice loss |
| `models.py` | EfficientNet-B0 U-Net (`effnet_unet`) used by the 0.90638 single-view baseline |
| `distillation.py` | Optional teacher–student distillation |

Why it matters: it keeps original pixels, validates on full images with the
official metric, and uses overlapping Hann-blended inference. An ordinary U-Net
with this recipe reaches 0.90638 without a strong encoder.

The `rle.py` helper used by this family is in `../common/rle.py`.
