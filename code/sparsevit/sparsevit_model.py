"""Adapter for the official SparseViT (AAAI2025) manipulation-localization model.

Returns raw logits so it plugs into the same training loop as the smp models.
Weights can come from ``checkpoint_path`` (official full-model checkpoint).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import SparseViT_Mul as _svt_module

# Fixed insertion order of the encoder's multi-level feature dictionary.
FEATURE_KEYS = ("third1", "third2", "third3", "third", "last1", "last")


class SparseViTSegmentation(nn.Module):
    def __init__(self, img_size: int = 512, pretrained_path=None,
                 checkpoint_path=None, drop_path: float = 0.2,
                 s_blocks3=None, s_blocks4=None):
        super().__init__()
        # SparseViT_Mul hardcodes drop_path_rate=0.2 when it calls SparseViT().
        # Patch the class reference *inside that module* so the constructor
        # receives our value, without forking the official repo.
        _orig_cls = _svt_module.SparseViT  # bound by `from SparseViT import SparseViT`
        if drop_path != _orig_default_rate():
            def _patched(*a, **kw):
                kw["drop_path_rate"] = drop_path
                return _orig_cls(*a, **kw)
            _svt_module.SparseViT = _patched
        try:
            mul_kw = dict(img_size=img_size, pretrained_path=pretrained_path)
            if s_blocks3 is not None:
                mul_kw["s_blocks3"] = s_blocks3
            if s_blocks4 is not None:
                mul_kw["s_blocks4"] = s_blocks4
            self.net = _svt_module.SparseViT_Mul(**mul_kw)
        finally:
            _svt_module.SparseViT = _orig_cls
        if checkpoint_path is not None:
            ck = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
            state = ck.get("model", ck) if isinstance(ck, dict) else ck
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            print(f"[sparsevit] img_size={img_size} drop_path={drop_path} "
                  f"loaded checkpoint: missing={len(missing)} "
                  f"unexpected={len(unexpected)}", flush=True)
            del ck, state

    def forward(self, x):
        feats = self.net.encoder_net(x)
        logits = self.net.lmu([feats[k] for k in FEATURE_KEYS])
        if tuple(logits.shape[-2:]) != tuple(x.shape[-2:]):
            logits = F.interpolate(logits, size=x.shape[-2:],
                                   mode="bilinear", align_corners=False)
        return logits


def _orig_default_rate() -> float:
    return 0.2
