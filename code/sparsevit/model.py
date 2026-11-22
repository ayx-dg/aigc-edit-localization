"""Segmentation models for the AIGC-edit localization experiments."""
from __future__ import annotations

import sys
from pathlib import Path

import segmentation_models_pytorch as smp

ARCH_ENCODERS = {
    "unet": "resnet34",
    "unetpp": "resnet34",
    "segformer": "mit_b0",
    "deeplabv3plus": "resnet50",
    # EfficientNet-B0 native-512 baseline from a prior Kaggle run (0.906).
    "effnet_unet": "efficientnet-b0",
}


def build_model(encoder_weights: str = "imagenet", arch: str = "unet",
                img_size: int = 512, drop_path: float = 0.2,
                s_blocks3=None, s_blocks4=None):
    """Build one of the small models used for the comparison experiments.

    ``encoder_weights`` can be "imagenet" (downloaded/cached by smp) or a
    path to a local state_dict file.
    ``img_size`` and ``drop_path`` only affect the SparseViT architecture.
    """
    if arch == "sparsevit":
        # Official AAAI2025 SparseViT repo, included under ./SparseViT.
        repo = Path(__file__).resolve().parent / "SparseViT"
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from sparsevit_model import SparseViTSegmentation
        # "imagenet" -> random init; any other value is a full-model checkpoint.
        ckpt = None if encoder_weights in ("imagenet", None) else encoder_weights
        return SparseViTSegmentation(img_size=img_size, checkpoint_path=ckpt,
                                     drop_path=drop_path,
                                     s_blocks3=s_blocks3, s_blocks4=s_blocks4)

    if arch not in ARCH_ENCODERS:
        raise ValueError(f"unknown arch {arch!r}; choose from {list(ARCH_ENCODERS)}")
    kwargs = dict(
        encoder_name=ARCH_ENCODERS[arch],
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,  # raw logits; sigmoid is applied in inference / metrics
    )
    if arch in ("unet", "effnet_unet"):
        return smp.Unet(**kwargs)
    if arch == "unetpp":
        return smp.UnetPlusPlus(**kwargs)
    if arch == "segformer":
        return smp.Segformer(**kwargs)
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(**kwargs)
    raise AssertionError("unreachable")
