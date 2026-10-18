"""Predict W/A/S/D for a folder of frames (or a CSV with a `filename` column).

usage:
  python predict.py --images ../DATA-QUEST-R02-SDC-GTA-VICE-CITY-master/test_images \
                    --csv ../DATA-QUEST-R02-SDC-GTA-VICE-CITY-master/test.csv --out submission.csv
  python predict.py --images /any/folder/of/frames --out preds.csv
"""
import argparse
import os

import pandas as pd
import torch

from common import KEYS, WASDModel, list_images

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='folder of frames, a single image, or a CSV with filename column')
    ap.add_argument('--csv', help='optional CSV whose `filename` column fixes which files / order to predict')
    ap.add_argument('--model', default=os.path.join(HERE, 'model.pt'))
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--no-tta', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    if not os.path.isfile(args.model):
        raise SystemExit(f'model file not found: {args.model} (pass --model path/to/model.pt)')

    if args.csv:
        names = pd.read_csv(args.csv)['filename'].astype(str).tolist()
        base = args.images if os.path.isdir(args.images) else os.path.dirname(args.images)
        paths = [os.path.join(base, n) for n in names]
    else:
        paths = list_images(args.images)
        names = [os.path.basename(p) for p in paths]
    if not paths:
        raise SystemExit(f'no images found at {args.images}')

    model = WASDModel.load(args.model)
    wasd, p_lon, p_lat, ok = model.predict_paths(paths, tta=not args.no_tta, log_every=5)
    out = pd.DataFrame(wasd, columns=KEYS)
    out.insert(0, 'filename', names)
    out.to_csv(args.out, index=False)
    print(f'wrote {args.out}: {len(out)} rows, {(~ok).sum()} unreadable')
    print('predicted key rates:', {k: round(float(out[k].mean()), 3) for k in KEYS})


if __name__ == '__main__':
    main()
