"""Small U-Net, strict column-major RLE, data loading and official metrics."""
import csv
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'AIGC Detection Dataset 2026'


def rows(path):
    csv.field_size_limit(100_000_000)
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def decode(text, height, width):
    flat = np.zeros(height * width, np.uint8)
    values = np.fromstring(text.strip(), dtype=np.int64, sep=' ') if text.strip() else np.array([], dtype=np.int64)
    if len(values) % 2:
        raise ValueError('RLE must contain start/length pairs')
    end = 0
    for start, length in values.reshape(-1, 2):
        if start < 1 or length < 1 or start-1 < end or start-1+length > flat.size:
            raise ValueError('Invalid RLE range')
        flat[start-1:start-1+length] = 1
        end = start-1+length
    return flat.reshape((height, width), order='F')


def encode(mask):
    flat = np.asarray(mask, dtype=np.uint8).ravel(order='F')
    edges = np.flatnonzero(np.diff(np.concatenate(([0], flat, [0]))))
    pairs = np.column_stack((edges[::2]+1, edges[1::2]-edges[::2]))
    return ' '.join(map(str, pairs.ravel()))


def split_rows(records, seed=42, val_fraction=.1):
    # Stratify by original edited area; test data never enter this split.
    groups = {}
    for r in records:
        area = sum(map(int, r['annotation'].split()[1::2]))
        ratio = area / (int(r['height'])*int(r['width']))
        bucket = 0 if area == 0 else 1+min(9, int(ratio*10))
        groups.setdefault(bucket, []).append(r)
    rng = random.Random(seed)
    train, val = [], []
    for key in sorted(groups):
        group = groups[key][:]
        rng.shuffle(group)
        n = min(len(group)-1, max(1, round(len(group)*val_fraction)))
        val.extend(group[:n]); train.extend(group[n:])
    return train, val


def tensor_image(image, size):
    array = np.array(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)/255
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class Images(Dataset):
    def __init__(self, records, data, size, augment=False):
        self.records, self.data, self.size, self.augment = records, Path(data), size, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(self.data/'train'/'images'/f"{r['ID']}.webp") as im:
            image = im.convert('RGB')
        h,w = int(r['height']), int(r['width'])
        if image.size != (w,h):
            raise ValueError(f"Image dimensions differ from CSV: {r['ID']}")
        mask = Image.fromarray(decode(r['annotation'], h,w))
        image = image.resize((self.size,self.size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.size,self.size), Image.Resampling.NEAREST)
        if self.augment:
            if random.random() < .5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            if random.random() < .5:
                image, mask = ImageOps.flip(image), ImageOps.flip(mask)
            k = random.randrange(4)
            image, mask = image.rotate(90*k), mask.rotate(90*k)
        return tensor_image(image,self.size), torch.from_numpy(np.array(mask,dtype=np.float32)[None])


def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(8,b),nn.ReLU(),
                         nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(8,b),nn.ReLU())


class UNet(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.down = nn.ModuleList([block(3,width),block(width,width*2),block(width*2,width*4),block(width*4,width*8)])
        self.up = nn.ModuleList([block(width*12,width*4),block(width*6,width*2),block(width*3,width)])
        self.head = nn.Conv2d(width,1,1)

    def forward(self,x):
        skips=[]
        for i, layer in enumerate(self.down):
            x=layer(x)
            if i<3:
                skips.append(x); x=F.max_pool2d(x,2)
        for layer,skip in zip(self.up,reversed(skips)):
            x=layer(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
        return self.head(x)


def loss_fn(logits, target, dice_weight=1.):
    bce=F.binary_cross_entropy_with_logits(logits,target)
    p=logits.sigmoid()
    dice=(2*(p*target).sum((1,2,3))+1)/(p.sum((1,2,3))+target.sum((1,2,3))+1)
    return bce+dice_weight*(1-dice).mean()


def metrics(pred, truth):
    pred,truth=np.asarray(pred,bool),np.asarray(truth,bool)
    a,b=pred.sum(),truth.sum()
    if a==0 and b==0: return 1.,1.
    if a==0 or b==0: return 0.,0.
    hit=np.logical_and(pred,truth).sum()
    return float(hit/(a+b-hit)),float(hit/b)


@torch.no_grad()
def probability(model,image,size,device):
    h,w=image.height,image.width
    x=tensor_image(image,size).unsqueeze(0).to(device)
    p=model(x).sigmoid()
    return F.interpolate(p,size=(h,w),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
