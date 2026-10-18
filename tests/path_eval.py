"""Honest path-following evaluation from out-of-fold predictions (every frame predicted by a model that never saw it).

Routes are built from consecutive training frames; the model's controls and the true controls each drive the
simulated car, and we measure how far apart the two paths end up. Compared against trivial drivers.
usage: python tests/path_eval.py --oof oof_convnext_tiny.csv [--route-len 30]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import KEYS  # noqa: E402
from simulate import simulate  # noqa: E402


def route_scores(pred, truth, L):
    out = []
    for s in range(0, len(truth) - L + 1, L):
        tp, tt = simulate(pred[s:s + L]), simulate(truth[s:s + L])
        dev = np.linalg.norm(tp[:, :2] - tt[:, :2], axis=1)
        plen = np.sum(np.linalg.norm(np.diff(tt[:, :2], axis=0), axis=1)) or 1
        heading_err = np.degrees(np.abs(np.angle(np.exp(1j * (tp[-1, 2] - tt[-1, 2])))))
        out.append((dev.mean(), 100 * dev[-1] / plen, heading_err))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', required=True)
    ap.add_argument('--route-len', type=int, default=30)
    a = ap.parse_args()
    df = pd.read_csv(a.oof)
    df['id'] = df.path.map(lambda p: int(os.path.basename(p).split('_')[0]))
    df = df.sort_values('id')
    truth = df[KEYS].values.astype(int)
    rng = np.random.RandomState(0)
    drivers = {
        'MODEL (out-of-fold)': df[[k + '_pred' for k in KEYS]].values.astype(int),
        'always W': np.tile([1, 0, 0, 0], (len(df), 1)),
        'random w/ label priors': truth[rng.permutation(len(truth))],
    }
    print(f'{len(df)} frames, routes of {a.route_len} frames')
    print(f'{"driver":<26}{"mean dev":>10}{"final err % path":>18}{"heading err deg":>17}{"exact":>8}')
    for name, p in drivers.items():
        r = route_scores(p, truth, a.route_len)
        print(f'{name:<26}{r[:, 0].mean():>10.1f}{np.median(r[:, 1]):>18.1f}{np.median(r[:, 2]):>17.1f}{(p == truth).all(1).mean():>8.3f}')


if __name__ == '__main__':
    main()
