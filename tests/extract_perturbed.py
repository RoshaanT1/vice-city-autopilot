"""Extract backbone features (original + mirrored) for degraded copies of the clean training frames.
usage: python tests/extract_perturbed.py <perturbation> --data <repo folder>
perturbations: blur, gray, jpeg, lowres
"""
import argparse
import io
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SOL)
from common import MEAN, STD, build_backbone, load_rgb, views  # noqa: E402


def jpeg(im, q=40):
    b = io.BytesIO()
    im.save(b, 'JPEG', quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert('RGB')


PERTURB = {
    'blur': lambda im: im.filter(ImageFilter.GaussianBlur(1.5)),
    'gray': lambda im: im.convert('L').convert('RGB'),
    'jpeg': jpeg,
    'lowres': lambda im: im.resize((480, 270), Image.BILINEAR).resize(im.size, Image.BILINEAR),
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('perturbation', choices=sorted(PERTURB))
    ap.add_argument('--data', required=True)
    ap.add_argument('--threads', type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    df = pd.read_csv(os.path.join(SOL, 'train_clean.csv'))
    paths = [os.path.join(a.data, 'train_images', f) for f in df.filename]
    bb = build_backbone('efficientnet_b0')
    fn = PERTURB[a.perturbation]
    x, xf = [], []
    for i in range(0, len(paths), 8):
        fulls, cars = [], []
        for p in paths[i:i + 8]:
            f, c = views(fn(load_rgb(p)))
            fulls.append(f)
            cars.append(c)
        F, Cc = torch.stack(fulls), torch.stack(cars)
        for flip, out in ((False, x), (True, xf)):
            f_, c_ = (F.flip(-1), Cc.flip(-1)) if flip else (F, Cc)
            out.append(torch.cat([bb((f_ - MEAN) / STD), bb((c_ - MEAN) / STD)], 1).numpy())
        if i % 80 == 0:
            print(a.perturbation, i, flush=True)
    np.savez(os.path.join(SOL, f'.feat_pert_{a.perturbation}.npz'), x=np.concatenate(x), xf=np.concatenate(xf))
    print(a.perturbation, 'done', flush=True)


if __name__ == '__main__':
    main()
