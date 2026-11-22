"""Original-pixel crops and overlapping full-image inference. No image resizing."""
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .core import decode, metrics


def crop_pair(image, mask, size, rng):
    h,w=mask.shape
    # Half the crops are centered near a random edited pixel, half uniform.
    # Sampling RLE/foreground is train-only; validation covers every pixel.
    if rng.random()<.5 and mask.any():
        positions=np.flatnonzero(mask)
        y,x=divmod(int(positions[rng.randrange(len(positions))]),w)
        top=max(0,min(y-rng.randrange(size),max(0,h-size)))
        left=max(0,min(x-rng.randrange(size),max(0,w-size)))
    else:
        top=rng.randrange(max(1,h-size+1));left=rng.randrange(max(1,w-size+1))
    im=image[top:top+size,left:left+size]
    target=mask[top:top+size,left:left+size]
    pad=((0,size-im.shape[0]),(0,size-im.shape[1]))
    # Edge replication only for small images, keeping labels aligned.
    return np.pad(im,(*pad,(0,0)),mode='edge'),np.pad(target,pad,mode='edge')


class NativeCrops(Dataset):
    def __init__(self,records,data,size):
        self.records,self.data,self.size=records,Path(data),size

    def __len__(self): return len(self.records)

    def __getitem__(self,index):
        r=self.records[index];h,w=int(r['height']),int(r['width'])
        with Image.open(self.data/'train/images'/f"{r['ID']}.webp") as im:
            image=np.array(im.convert('RGB'))
        if image.shape[:2]!=(h,w): raise ValueError('CSV dimensions mismatch')
        image,mask=crop_pair(image,decode(r['annotation'],h,w),self.size,random)
        k=random.randrange(4)
        image,mask=np.rot90(image,k),np.rot90(mask,k)
        if random.random()<.5: image,mask=image[:,::-1],mask[:,::-1]
        return torch.from_numpy(image.transpose(2,0,1).copy()).float()/255,torch.from_numpy(mask[None].copy()).float()


def starts(length,size,stride):
    return list(range(0,max(0,length-size),stride))+[max(0,length-size)]


@torch.no_grad()
def native_probability(model,image,size,device,overlap=64,tta=1):
    if not 0<=overlap<size: raise ValueError('overlap must be smaller than tile size')
    if tta not in (1,4): raise ValueError('tta must be 1 or 4')
    model.eval()
    raw=np.array(image.convert('RGB'));h,w=raw.shape[:2]
    total=np.zeros((h,w),np.float32);weight=np.zeros((h,w),np.float32)
    # Nonzero tapered weighting reduces seams without leaving border holes.
    axis=np.maximum(np.hanning(size),.05).astype(np.float32)
    window=axis[:,None]*axis[None,:]
    for y in starts(h,size,size-overlap):
        for x in starts(w,size,size-overlap):
            crop=raw[y:y+size,x:x+size];ch,cw=crop.shape[:2]
            crop=np.pad(crop,((0,size-ch),(0,size-cw),(0,0)),mode='edge')
            tensor=torch.from_numpy(crop.transpose(2,0,1).copy()).float().unsqueeze(0).to(device)/255
            with torch.cuda.amp.autocast(enabled=str(device).startswith('cuda')):
                # Invert each flip BEFORE combining probabilities or cropping padding.
                transforms=[()] if tta==1 else [(),(2,),(3,),(2,3)]
                combined=None
                for dims in transforms:
                    transformed=torch.flip(tensor,dims) if dims else tensor
                    q=model(transformed).float().sigmoid()
                    if dims: q=torch.flip(q,dims)
                    combined=q if combined is None else combined+q
                p=(combined/tta)[0,0,:ch,:cw].cpu().numpy()
            total[y:y+ch,x:x+cw]+=p*window[:ch,:cw]
            weight[y:y+ch,x:x+cw]+=window[:ch,:cw]
    return total/weight


def threshold_metrics(probability, truth, thresholds):
    """Exact per-image threshold sweep, without materializing K full masks."""
    # Match NumPy's float32 mask comparison semantics at exact boundaries.
    bins=np.searchsorted(np.asarray(thresholds,dtype=probability.dtype),probability.ravel(),side='left')
    positive=np.asarray(truth,bool).ravel()
    total=np.bincount(bins,minlength=len(thresholds)+1)
    hits=np.bincount(bins[positive],minlength=len(thresholds)+1)
    predicted=np.cumsum(total[::-1])[::-1][1:]
    tp=np.cumsum(hits[::-1])[::-1][1:]
    gt=positive.sum()
    iou=tp/np.maximum(predicted+gt-tp,1)
    recall=tp/max(gt,1)
    empty=(predicted==0)&(gt==0)
    iou[empty]=1.;recall[empty]=1.
    return np.stack([iou,recall],axis=1)


def validate_native(model,records,data,size,device,overlap,thresholds=None):
    thresholds=thresholds if thresholds is not None else [.1,.2,.3,.4,.5,.6,.7,.8,.9]
    totals=np.zeros((len(thresholds),2))
    for i,r in enumerate(records):
        with Image.open(Path(data)/'train/images'/f"{r['ID']}.webp") as im:
            p=native_probability(model,im,size,device,overlap)
        truth=decode(r['annotation'],int(r['height']),int(r['width']))
        totals+=threshold_metrics(p,truth,thresholds)
        if i%25==0: print(f'Full-resolution validation {i+1}/{len(records)}',flush=True)
    totals/=len(records);scores=.7*totals[:,0]+.3*totals[:,1];best=int(scores.argmax())
    return dict(miou=float(totals[best,0]),mrecall=float(totals[best,1]),score=float(scores[best]),threshold=thresholds[best])
