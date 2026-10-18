"""Clean the training data, cross-validate, and fit the final WASD model.

usage: python train.py --data <repo folder with train.csv + train_images/> [--backbone convnext_tiny]
"""
import argparse
import hashlib
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import (KEYS, WASDModel, axes_to_wasd, build_backbone, extract, labels_to_axes,
                    load_rgb, swap_lat_probs)

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- cleaning
def clean(data_dir):
    """Return a cleaned dataframe (path, W, A, S, D) and a report of every row that was dropped or fixed."""
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    img_dir = os.path.join(data_dir, 'train_images')
    on_disk = set(os.listdir(img_dir))
    report = []

    # 1) CSV names that do not exist on disk: the same frames were saved under a different id prefix but
    #    keep the same _WASD suffix, so re-link each one to an unreferenced file with the identical suffix.
    unref = sorted(on_disk - set(df.filename))
    suffix = lambda n: n.split('_', 1)[-1]
    for i, row in df[~df.filename.isin(on_disk)].iterrows():
        match = next((u for u in unref if suffix(u) == suffix(row.filename)), None)
        if match:
            unref.remove(match)
            report.append((row.filename, 'relinked', match))
            df.at[i, 'filename'] = match
        else:
            report.append((row.filename, 'dropped: file missing', ''))
    df = df[df.filename.isin(on_disk)].copy()

    keep = []
    seen = {}
    for _, row in df.iterrows():
        name, lab = row.filename, row[KEYS].astype(int).tolist()
        path = os.path.join(img_dir, name)
        # 2) off-distribution images: the .jpg files are web screenshots from other GTA games at random sizes
        img = load_rgb(path)
        if img is None:
            report.append((name, 'dropped: unreadable', ''))
            continue
        if not name.lower().endswith('.png') or img.size != (1366, 768):
            report.append((name, f'dropped: not a Vice City frame ({img.size[0]}x{img.size[1]} {name.rsplit(".", 1)[-1]})', ''))
            continue
        # 3) physically contradictory controls
        if (lab[0] and lab[2]) or (lab[1] and lab[3]):
            report.append((name, 'dropped: contradictory label (W+S or A+D)', ''))
            continue
        # 4) filename suffix encodes WASD for genuine frames; flag disagreement with the CSV
        suf = os.path.splitext(name)[0].split('_')[-1]
        if len(suf) == 4 and set(suf) <= {'0', '1'} and [int(c) for c in suf] != lab:
            report.append((name, f'label mismatch csv={lab} filename={suf}; kept csv', ''))
        # 5) exact duplicates
        h = hashlib.md5(open(path, 'rb').read()).hexdigest()
        if h in seen:
            report.append((name, 'dropped: duplicate of ' + seen[h], ''))
            continue
        seen[h] = name
        keep.append([path] + lab)
    clean_df = pd.DataFrame(keep, columns=['path'] + KEYS)
    return clean_df, pd.DataFrame(report, columns=['filename', 'action', 'detail'])


# ----------------------------------------------------------------------------- modelling
def fit_heads(x, lon, lat, xf, C, cw):
    """Fit scaler + two heads on original features plus mirrored copies (A<->D swapped)."""
    lat_flip = np.where(lat == 1, 2, np.where(lat == 2, 1, 0))
    X = np.vstack([x, xf])
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    mk = lambda: LogisticRegression(C=C, class_weight=cw, max_iter=3000)
    h_lon = mk().fit(Xs, np.concatenate([lon, lon]))
    h_lat = mk().fit(Xs, np.concatenate([lat, lat_flip]))
    return scaler, h_lon, h_lat


def predict_probs(scaler, h_lon, h_lat, x, xf):
    p_lon = (h_lon.predict_proba(scaler.transform(x)) + h_lon.predict_proba(scaler.transform(xf))) / 2
    p_lat = (h_lat.predict_proba(scaler.transform(x)) + swap_lat_probs(h_lat.predict_proba(scaler.transform(xf)))) / 2
    return p_lon, p_lat


def metrics(y, yhat):
    out = {k: {'acc': float((y[:, i] == yhat[:, i]).mean()),
               'f1': float(f1_score(y[:, i], yhat[:, i], zero_division=0))} for i, k in enumerate(KEYS)}
    out['macro_f1'] = float(np.mean([out[k]['f1'] for k in KEYS]))
    out['exact_match'] = float((y == yhat).all(1).mean())
    out['mean_key_acc'] = float((y == yhat).mean())
    return out


BIAS_GRID = np.arange(-1.5, 1.51, 0.25)


def decide(p_lon, p_lat, b_lon, b_lat):
    lon = np.argmax(np.log(p_lon + 1e-9) + b_lon, 1)
    lat = np.argmax(np.log(p_lat + 1e-9) + b_lat, 1)
    return axes_to_wasd(lon, lat)


def tune_bias(p_lon, p_lat, y):
    """Grid-search additive log-biases for classes 1 and 2 of each axis (axes are independent)."""
    best = {}
    for axis, p, cols in (('lon', p_lon, [0, 2]), ('lat', p_lat, [1, 3])):
        scores = []
        for b1, b2 in itertools.product(BIAS_GRID, BIAS_GRID):
            b = np.array([0, b1, b2])
            pred = np.argmax(np.log(p + 1e-9) + b, 1)
            onehot = np.stack([pred == 1, pred == 2], 1).astype(int)
            s = np.mean([f1_score(y[:, c], onehot[:, j], zero_division=0) for j, c in enumerate(cols)])
            scores.append((s, -abs(b1) - abs(b2), b))  # tie-break toward small bias
        best[axis] = max(scores, key=lambda t: (t[0], t[1]))[2]
    return best['lon'], best['lat']


def honest_bias_eval(p_lon, p_lat, y, seed=0):
    """Tune biases on one half of the OOF predictions and score on the other half, both ways."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    halves = [idx[: len(y) // 2], idx[len(y) // 2:]]
    yhat = np.zeros_like(y)
    for a, b in ((0, 1), (1, 0)):
        bl, bt = tune_bias(p_lon[halves[a]], p_lat[halves[a]], y[halves[a]])
        yhat[halves[b]] = decide(p_lon[halves[b]], p_lat[halves[b]], bl, bt)
    return yhat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--backbone', default='convnext_tiny')
    ap.add_argument('--out', default=os.path.join(HERE, 'model.pt'))
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--Cs', default='0.1', help='comma-separated C grid')
    ap.add_argument('--cws', default='balanced', help='comma-separated class_weight grid: none,balanced')
    ap.add_argument('--tune-bias', action='store_true',
                    help='tune per-class decision biases (off: held-out eval showed it overfits and worsens paths)')
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    t0 = time.time()

    df, report = clean(args.data)
    report.to_csv(os.path.join(HERE, 'cleaning_report.csv'), index=False)
    df.assign(filename=df.path.map(os.path.basename))[['filename'] + KEYS].to_csv(os.path.join(HERE, 'train_clean.csv'), index=False)
    print(f'clean rows: {len(df)}  | report:\n{report.action.str.split(":").str[0].value_counts().to_string()}')
    y = df[KEYS].values.astype(int)
    lon, lat = labels_to_axes(y)
    print('label combos after cleaning:', df[KEYS].astype(str).agg(''.join, axis=1).value_counts().to_dict())

    cache = os.path.join(HERE, f'.feat_cache_{args.backbone}.npz')
    backbone = build_backbone(args.backbone)
    if os.path.exists(cache) and list(np.load(cache, allow_pickle=True)['paths']) == df.path.tolist():
        z = np.load(cache, allow_pickle=True)
        x, xf = z['x'], z['xf']
        print('loaded cached features', x.shape)
    else:
        print('extracting features...')
        x, _ = extract(backbone, df.path.tolist(), log_every=10)
        xf, _ = extract(backbone, df.path.tolist(), flip=True)
        np.savez(cache, x=x, xf=xf, paths=np.array(df.path.tolist(), dtype=object))
        print(f'features {x.shape} in {time.time() - t0:.0f}s')

    strata = lon * 3 + lat
    # merge strata with <2 members per fold into a neighbour so StratifiedKFold works
    counts = pd.Series(strata).value_counts()
    strata = np.array([s if counts[s] >= 5 else 99 for s in strata])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=args.repeats, random_state=42)

    results = []
    baseline = metrics(y, np.tile([1, 0, 0, 0], (len(y), 1)))
    print(f'baseline always-W: macro_f1={baseline["macro_f1"]:.3f} exact={baseline["exact_match"]:.3f} key_acc={baseline["mean_key_acc"]:.3f}')
    grid = itertools.product([float(c) for c in args.Cs.split(',')],
                             [None if c == 'none' else c for c in args.cws.split(',')])
    for C, cw in grid:
        oof_lon = np.zeros((args.repeats, len(y), 3))
        oof_lat = np.zeros((args.repeats, len(y), 3))
        for k, (tr, va) in enumerate(cv.split(x, strata)):
            s, hl, ht = fit_heads(x[tr], lon[tr], lat[tr], xf[tr], C, cw)
            pl, pt = predict_probs(s, hl, ht, x[va], xf[va])
            oof_lon[k // 5, va], oof_lat[k // 5, va] = pl, pt
        raw = [metrics(y, decide(oof_lon[r], oof_lat[r], 0, 0)) for r in range(args.repeats)]
        tuned = [metrics(y, honest_bias_eval(oof_lon[r], oof_lat[r], y, seed=r)) for r in range(args.repeats)]
        agg = lambda ms, key: float(np.mean([m[key] for m in ms]))
        res = dict(C=C, cw=cw or 'none', raw_macro_f1=agg(raw, 'macro_f1'), raw_exact=agg(raw, 'exact_match'),
                   tuned_macro_f1=agg(tuned, 'macro_f1'), tuned_exact=agg(tuned, 'exact_match'),
                   tuned_key_acc=agg(tuned, 'mean_key_acc'), std_tuned_f1=float(np.std([m['macro_f1'] for m in tuned])),
                   per_key={kk: {m: float(np.mean([t[kk][m] for t in tuned])) for m in ('acc', 'f1')} for kk in KEYS})
        results.append((res, oof_lon.mean(0), oof_lat.mean(0)))
        print(f'C={C:<6} cw={res["cw"]:<8} raw f1={res["raw_macro_f1"]:.3f} exact={res["raw_exact"]:.3f} | '
              f'bias-tuned (held-out) f1={res["tuned_macro_f1"]:.3f}±{res["std_tuned_f1"]:.3f} exact={res["tuned_exact"]:.3f} '
              f'keyacc={res["tuned_key_acc"]:.3f}  ' + ' '.join(f'{kk}:{res["per_key"][kk]["f1"]:.2f}' for kk in KEYS), flush=True)

    best, b_oof_lon, b_oof_lat = max(results, key=lambda r: r[0]['tuned_macro_f1' if args.tune_bias else 'raw_macro_f1'])
    bias_lon, bias_lat = tune_bias(b_oof_lon, b_oof_lat, y) if args.tune_bias else (np.zeros(3), np.zeros(3))
    print(f'\nBEST: C={best["C"]} cw={best["cw"]} bias_lon={bias_lon.tolist()} bias_lat={bias_lat.tolist()}')
    print(json.dumps(best, indent=1))

    s, hl, ht = fit_heads(x, lon, lat, xf, best['C'], None if best['cw'] == 'none' else best['cw'])
    model = WASDModel(args.backbone, backbone, s, hl, ht, bias_lon, bias_lat)
    model.save(args.out)
    pd.DataFrame(np.hstack([df[['path']].values, y, decide(b_oof_lon, b_oof_lat, bias_lon, bias_lat)]),
                 columns=['path'] + KEYS + [k + '_pred' for k in KEYS]).to_csv(os.path.join(HERE, f'oof_{args.backbone}.csv'), index=False)
    with open(os.path.join(HERE, f'cv_results_{args.backbone}.json'), 'w') as f:
        json.dump({'baseline_always_W': baseline, 'grid': [r[0] for r in results], 'best': best,
                   'bias_lon': bias_lon.tolist(), 'bias_lat': bias_lat.tolist()}, f, indent=1)
    print(f'saved {args.out}  total {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
