"""Aligned original-pixel RGB, ground truth and offline teacher probabilities."""
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .core import decode
from .native import crop_pair


class DistillationCrops(torch.utils.data.Dataset):
    def __init__(self,records,data,size,cache):
        self.records,self.data,self.size,self.cache=records,Path(data),size,Path(cache)

    def __len__(self): return len(self.records)

    def __getitem__(self,index):
        r=self.records[index];h,w=int(r['height']),int(r['width'])
        with Image.open(self.data/'train/images'/f"{r['ID']}.webp") as im: rgb=np.array(im.convert('RGB'))
        soft=np.load(self.cache/f"{r['ID']}.npy",mmap_mode='r')
        if soft.shape!=(h,w) or rgb.shape[:2]!=(h,w): raise ValueError('Soft target dimensions mismatch')
        # A fourth channel guarantees exactly the same crop/padding/augmentation.
        packed=np.concatenate([rgb.astype(np.float32),soft[:,:,None].astype(np.float32)],axis=2)
        packed,hard=crop_pair(packed,decode(r['annotation'],h,w),self.size,random)
        k=random.randrange(4);packed,hard=np.rot90(packed,k),np.rot90(hard,k)
        if random.random()<.5: packed,hard=packed[:,::-1],hard[:,::-1]
        return (torch.from_numpy(packed[:,:,:3].transpose(2,0,1).copy())/255,
                torch.from_numpy(hard[None].copy()).float(),
                torch.from_numpy(packed[:,:,3][None].copy()))


def soft_loss(logits,soft,hard):
    # Ground-truth agreement reduces influence of confidently wrong teacher pixels.
    weight=(hard*soft+(1-hard)*(1-soft)).detach().clamp_min(.1)
    return (F.binary_cross_entropy_with_logits(logits,soft,reduction='none')*weight).sum()/weight.sum().clamp_min(1)
