"""Checkpoint-driven model factory. All models accept RGB floats in [0,1]."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from .core import ROOT,UNet

WEIGHTS=ROOT/'weights/efficientnet-b0-355c32eb.pth'


class PretrainedUNet(nn.Module):
    def __init__(self,encoder='efficientnet-b0',initialize=False):
        super().__init__()
        import segmentation_models_pytorch as smp
        if encoder!='efficientnet-b0': raise ValueError('Only audited B0 configuration supported')
        self.network=smp.Unet(encoder_name=encoder,encoder_weights=None,in_channels=3,classes=1)
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('rgb_std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        if initialize:
            if not WEIGHTS.exists(): raise FileNotFoundError('Run prepare_pretrained.py first')
            digest=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
            if not digest.startswith('355c32eb'): raise ValueError('ImageNet weight hash mismatch')
            state=torch.load(WEIGHTS,map_location='cpu')
            self.network.encoder.load_state_dict(state)

    def forward(self,x):
        return self.network((x-self.rgb_mean)/self.rgb_std)


def build_model(config,initialize=False):
    arch=config.get('arch','tiny_unet')
    if arch=='tiny_unet': return UNet(config.get('width',24))
    if arch=='smp_unet':
        if config.get('in_channels',3)!=3: raise ValueError('RGB baseline requires 3 channels')
        return PretrainedUNet(config.get('encoder','efficientnet-b0'),initialize)
    raise ValueError('Unknown architecture: '+arch)
