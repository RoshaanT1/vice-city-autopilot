"""Honest robustness: 5-fold CV where validation frames are degraded copies the model never saw.
Recipe A = train on clean frames only (shipped model). Recipe B = clean + blur/gray/jpeg/lowres copies.
usage: python tests/robustness_cv.py [--save-aug model_aug.pt]
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SOL)
sys.path.insert(0, os.path.join(SOL, 'tests'))
from common import KEYS, WASDModel, build_backbone, labels_to_axes  # noqa: E402
from path_eval import route_scores  # noqa: E402
from train import decide, fit_heads, metrics, predict_probs  # noqa: E402

PERTS = ['blur', 'gray', 'jpeg', 'lowres']
C, CW = 0.1, 'balanced'
df = pd.read_csv(os.path.join(SOL, 'train_clean.csv'))
y = df[KEYS].values
lon, lat = labels_to_axes(y)
z = np.load(os.path.join(SOL, '.feat_cache_efficientnet_b0.npz'), allow_pickle=True)
FEAT = {'clean': (z['x'], z['xf'])}
for p in PERTS:
    q = np.load(os.path.join(SOL, f'.feat_pert_{p}.npz'))
    FEAT[p] = (q['x'], q['xf'])
order = np.argsort([int(f.split('_')[0]) for f in df.filename])


def fit(recipe, idx):
    sets = ['clean'] if recipe == 'A' else ['clean'] + PERTS
    X = np.vstack([FEAT[s][0][idx] for s in sets])
    XF = np.vstack([FEAT[s][1][idx] for s in sets])
    k = len(sets)
    return fit_heads(X, np.tile(lon[idx], k), np.tile(lat[idx], k), XF, C, CW)


def run_fold(args):
    recipe, tr, va = args
    s, hl, ht = fit(recipe, tr)
    return recipe, va, {name: predict_probs(s, hl, ht, FEAT[name][0][va], FEAT[name][1][va]) for name in FEAT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save-aug')
    a = ap.parse_args()
    strata = lon * 3 + lat
    counts = pd.Series(strata).value_counts()
    strata = np.array([v if counts[v] >= 5 else 99 for v in strata])
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(y, strata))
    jobs = [(r, tr, va) for r in 'AB' for tr, va in folds]
    with Pool(6) as pool:
        res = pool.map(run_fold, jobs)
    oof = {(r, n): [np.zeros((len(y), 3)), np.zeros((len(y), 3))] for r in 'AB' for n in FEAT}
    for r, va, probs in res:
        for n, (pl, pt) in probs.items():
            oof[(r, n)][0][va], oof[(r, n)][1][va] = pl, pt
    print(f'{"recipe":<22}{"val frames":<10}{"macroF1":>8}{"exact":>7}{"keyacc":>7}{"A f1":>6}{"D f1":>6}{"S f1":>6}'
          f'{"path end%":>10}{"path dev":>9}{"same as clean":>14}')
    for r in 'AB':
        clean_dec = decide(*oof[(r, 'clean')], 0, 0)
        for n in FEAT:
            d = decide(*oof[(r, n)], 0, 0)
            m = metrics(y, d)
            rs = route_scores(d[order], y[order], 30)
            name = 'A clean-only (shipped)' if r == 'A' else 'B +degraded copies'
            print(f'{name:<22}{n:<10}{m["macro_f1"]:>8.3f}{m["exact_match"]:>7.3f}{m["mean_key_acc"]:>7.3f}'
                  f'{m["A"]["f1"]:>6.2f}{m["D"]["f1"]:>6.2f}{m["S"]["f1"]:>6.2f}{np.median(rs[:, 1]):>9.1f}%{rs[:, 0].mean():>9.1f}'
                  f'{(d == clean_dec).all(1).mean():>14.2f}', flush=True)
    # how confident were the decisions that flip under degradation?
    pl, pt = oof[('A', 'clean')]
    conf = np.minimum(pl.max(1), pt.max(1))
    for n in PERTS:
        flipped = (decide(*oof[('A', n)], 0, 0) != decide(pl, pt, 0, 0)).any(1)
        print(f'A/{n}: {flipped.mean():.0%} of frames flip; median confidence of flipped {np.median(conf[flipped]):.2f} '
              f'vs stable {np.median(conf[~flipped]):.2f}')
    if a.save_aug:
        s, hl, ht = fit('B', np.arange(len(y)))
        WASDModel('efficientnet_b0', build_backbone('efficientnet_b0'), s, hl, ht, np.zeros(3), np.zeros(3)).save(a.save_aug)
        print('saved', a.save_aug)


if __name__ == '__main__':
    main()
