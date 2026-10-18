"""Fast config sweep on cached features: 5-fold CV, reports key metrics and path-following vs always-W.
usage: python tests/sweep.py <featureset> <C> <cw>     featureset in {cx_full, cx_car, cx_both, ef_both, cx_ef}
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SOL)
sys.path.insert(0, os.path.join(SOL, 'tests'))
from common import KEYS, labels_to_axes  # noqa: E402
from train import decide, fit_heads, honest_bias_eval, metrics, predict_probs  # noqa: E402
from path_eval import route_scores  # noqa: E402

fs, C, cw = sys.argv[1], float(sys.argv[2]), (None if sys.argv[3] == 'none' else 'balanced')
df = pd.read_csv(os.path.join(SOL, 'train_clean.csv'))
y = df[KEYS].values
lon, lat = labels_to_axes(y)
cx = np.load(os.path.join(SOL, '.feat_cache_convnext_tiny.npz'), allow_pickle=True)
ef = np.load(os.path.join(SOL, '.feat_cache_efficientnet_b0.npz'), allow_pickle=True)
sel = {'cx_full': (cx, slice(0, 768)), 'cx_car': (cx, slice(768, 1536)), 'cx_both': (cx, slice(None)),
       'ef_both': (ef, slice(None)), 'ef_car': (ef, slice(1280, 2560))}
if fs == 'cx_ef':
    x, xf = np.hstack([cx['x'], ef['x']]), np.hstack([cx['xf'], ef['xf']])
else:
    z, s = sel[fs]
    x, xf = z['x'][:, s], z['xf'][:, s]
strata = lon * 3 + lat
counts = pd.Series(strata).value_counts()
strata = np.array([v if counts[v] >= 5 else 99 for v in strata])
pl, pt = np.zeros((len(y), 3)), np.zeros((len(y), 3))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(x, strata):
    s_, hl, ht = fit_heads(x[tr], lon[tr], lat[tr], xf[tr], C, cw)
    pl[va], pt[va] = predict_probs(s_, hl, ht, x[va], xf[va])
order = np.argsort([int(os.path.basename(f).split('_')[0]) for f in df.filename])
ids = df.filename.values
for tag, yhat in (('raw', decide(pl, pt, 0, 0)), ('bias', honest_bias_eval(pl, pt, y))):
    m = metrics(y, yhat)
    r = route_scores(yhat[order], y[order], 30)
    print(f'{fs:<8} C={C:<6} cw={sys.argv[3]:<8} {tag:<4} f1={m["macro_f1"]:.3f} exact={m["exact_match"]:.3f} keyacc={m["mean_key_acc"]:.3f} '
          + ' '.join(f'{k}:{m[k]["f1"]:.2f}' for k in KEYS) + f' | path dev={r[:, 0].mean():.1f} final%={np.median(r[:, 1]):.1f} head={np.median(r[:, 2]):.0f}', flush=True)
np.savez(os.path.join(SOL, f'.oof_{fs}_{C}_{sys.argv[3]}.npz'), pl=pl, pt=pt)
