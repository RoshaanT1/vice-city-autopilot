"""Stress tests for the WASD model, predict.py and simulate.py.

usage: python tests/stress_test.py --data ../DATA-QUEST-R02-SDC-GTA-VICE-CITY-master --work /tmp/stress
Every check prints PASS/FAIL; exit code is the number of failures.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SOL)
from common import KEYS, WASDModel, list_images  # noqa: E402
from simulate import simulate  # noqa: E402

PY = sys.executable
FAILS = []


def check(name, cond, info=''):
    print(f'[{"PASS" if cond else "FAIL"}] {name} {info}', flush=True)
    if not cond:
        FAILS.append(name)


def run(args, cwd=None, env=None, timeout=900):
    e = dict(os.environ, **(env or {}))
    r = subprocess.run([PY] + args, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=e)
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--n', type=int, default=24, help='test frames used for perturbation checks')
    a = ap.parse_args()
    data, work = os.path.abspath(a.data), os.path.abspath(a.work)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    model = WASDModel.load(os.path.join(SOL, 'model.pt'))
    test_dir = os.path.join(data, 'test_images')
    test_paths = list_images(test_dir)[: a.n]

    # ---------------------------------------------------------------- 1. perturbation invariance
    base, *_ = model.predict_paths(test_paths)
    perturb = {
        'resize_1920x1080': lambda im: im.resize((1920, 1080)),
        'resize_1280x720': lambda im: im.resize((1280, 720)),
        'resize_800x450': lambda im: im.resize((800, 450)),
        'stretch_1024x768_4:3': lambda im: im.resize((1024, 768)),
        'jpeg_q60': None,
        'grayscale': lambda im: im.convert('L'),
        'rgba': lambda im: im.convert('RGBA'),
        'darker_0.75': lambda im: ImageEnhance.Brightness(im).enhance(0.75),
        'brighter_1.25': lambda im: ImageEnhance.Brightness(im).enhance(1.25),
        'blur_r1.5': lambda im: im.filter(ImageFilter.GaussianBlur(1.5)),
    }
    for name, fn in perturb.items():
        d = os.path.join(work, 'perturb', name)
        os.makedirs(d)
        ps = []
        for p in test_paths:
            im = Image.open(p).convert('RGB')
            stem = os.path.splitext(os.path.basename(p))[0]
            if name == 'jpeg_q60':
                out = os.path.join(d, stem + '.jpg'); im.save(out, quality=60)
            else:
                out = os.path.join(d, stem + '.png'); fn(im).save(out)
            ps.append(out)
        pred, *_ = model.predict_paths(ps)
        agree = float((pred == base).all(1).mean())
        key_agree = float((pred == base).mean())
        # resolution / compression should barely change decisions; colour changes are allowed a bit more drift
        thr = 0.80 if name.startswith(('resize', 'jpeg', 'rgba')) else 0.6
        check(f'perturb {name}', agree >= thr, f'frame agreement={agree:.2f} key agreement={key_agree:.2f}')

    # ---------------------------------------------------------------- 2. messy input folder
    messy = os.path.join(work, 'messy')
    os.makedirs(os.path.join(messy, 'subdir'))
    for k, p in enumerate(test_paths[:12]):
        names = ['1.png', '2.png', '10.png', '3.PNG', 'frame 004.png', 'frame_ü5.png', '11.jpg', '20.jpeg',
                 '100.png', '0007.png', '12.png', '13.png']
        im = Image.open(p).convert('RGB')
        im.save(os.path.join(messy, names[k]), **({'quality': 90} if names[k].endswith(('jpg', 'jpeg')) else {}))
    open(os.path.join(messy, 'zero_bytes.png'), 'wb').close()
    with open(os.path.join(messy, 'truncated.png'), 'wb') as f:
        f.write(open(test_paths[0], 'rb').read()[:5000])
    with open(os.path.join(messy, 'not_an_image.png'), 'w') as f:
        f.write('hello')
    with open(os.path.join(messy, 'notes.txt'), 'w') as f:
        f.write('ignore me')
    shutil.copy(test_paths[0], os.path.join(messy, 'subdir', 'nested.png'))
    order = [os.path.basename(p) for p in list_images(messy)]
    check('natural sort order', order.index('2.png') < order.index('10.png') < order.index('100.png'), str(order))
    check('txt + subdir ignored', 'notes.txt' not in order and 'nested.png' not in order)
    rc, out = run([os.path.join(SOL, 'predict.py'), '--images', messy, '--out', os.path.join(work, 'messy_pred.csv')], cwd=work)
    ok = rc == 0 and os.path.exists(os.path.join(work, 'messy_pred.csv'))
    check('predict.py on messy folder (corrupt/zero-byte/non-image/unicode names)', ok, out.strip().splitlines()[-1] if out else '')
    if ok:
        mp = pd.read_csv(os.path.join(work, 'messy_pred.csv'))
        check('messy: one row per image file', len(mp) == len(order), f'{len(mp)} rows vs {len(order)} files')
        bad = mp[mp.filename.isin(['zero_bytes.png', 'not_an_image.png'])][KEYS].values
        check('messy: unreadable files fall back to W', len(bad) == 2 and (bad == [1, 0, 0, 0]).all(), str(bad.tolist()))

    # ---------------------------------------------------------------- 3. official submission + format
    sub = os.path.join(work, 'submission.csv')
    rc, out = run([os.path.join(SOL, 'predict.py'), '--images', test_dir, '--csv', os.path.join(data, 'test.csv'), '--out', sub], cwd=work)
    check('predict.py official test set', rc == 0, out.strip().splitlines()[-1] if out else '')
    if rc == 0:
        s, ss = pd.read_csv(sub), pd.read_csv(os.path.join(data, 'sample_submission.csv'))
        check('submission columns == sample', list(s.columns) == list(ss.columns), str(list(s.columns)))
        check('submission row order == sample', s.filename.tolist() == ss.filename.tolist())
        check('submission values binary', s[KEYS].isin([0, 1]).all().all())
        check('submission no W+S / A+D', not ((s.W & s.S) | (s.A & s.D)).any())

    # ---------------------------------------------------------------- 4. input modes
    rc, out = run([os.path.join(SOL, 'predict.py'), '--images', test_paths[0], '--out', os.path.join(work, 'single.csv')], cwd=work)
    check('predict.py single image', rc == 0)
    rc, out = run([os.path.join(SOL, 'simulate.py'), '--images', os.path.join(data, 'test.csv'), '--headless', '--out', os.path.join(work, 'sim_csv')], cwd=work)
    check('simulate.py with CSV input', rc == 0 and os.path.exists(os.path.join(work, 'sim_csv', 'path.png')))
    empty = os.path.join(work, 'empty'); os.makedirs(empty)
    rc, out = run([os.path.join(SOL, 'simulate.py'), '--images', empty, '--headless'], cwd=work)
    check('simulate.py empty folder exits cleanly with message', rc != 0 and 'no images' in out and 'Traceback' not in out)
    rc, out = run([os.path.join(SOL, 'simulate.py'), '--images', messy, '--headless', '--out', os.path.join(work, 'sim_messy')], cwd=work)
    check('simulate.py messy folder headless', rc == 0)

    # ---------------------------------------------------------------- 5. route with ground truth (labels in filenames)
    clean = pd.read_csv(os.path.join(SOL, 'train_clean.csv'))
    route = os.path.join(work, 'route_train')
    os.makedirs(route)
    for fn in clean.filename[:60]:
        shutil.copy(os.path.join(data, 'train_images', fn), route)
    rc, out = run([os.path.join(SOL, 'simulate.py'), '--images', route, '--headless', '--out', os.path.join(work, 'sim_route')], cwd=work)
    summ = json.load(open(os.path.join(work, 'sim_route', 'summary.json'))) if rc == 0 else {}
    check('simulate.py auto-detects truth from _WASD filenames', 'vs_truth' in summ, json.dumps(summ.get('vs_truth', {}).get('key_accuracy', {})))
    lab = os.path.join(work, 'labels.csv')
    clean.head(60).to_csv(lab, index=False)
    renamed = os.path.join(work, 'route_renamed'); os.makedirs(renamed)
    for k, fn in enumerate(clean.filename[:60]):
        shutil.copy(os.path.join(data, 'train_images', fn), os.path.join(renamed, fn))
    rc, out = run([os.path.join(SOL, 'simulate.py'), '--images', renamed, '--labels', lab, '--headless', '--out', os.path.join(work, 'sim_labels')], cwd=work)
    check('simulate.py with --labels CSV', rc == 0 and 'vs_truth' in json.load(open(os.path.join(work, 'sim_labels', 'summary.json'))))

    # ---------------------------------------------------------------- 6. real window loop (dummy video driver)
    rc, out = run([os.path.join(SOL, 'simulate.py'), '--images', route, '--fps', '60', '--autoclose', '--out', os.path.join(work, 'sim_window')],
                  cwd=work, env={'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy'})
    check('simulate.py window loop runs to the end and saves path.png', rc == 0 and os.path.exists(os.path.join(work, 'sim_window', 'path.png')), out.strip().splitlines()[-1] if out else '')

    # ---------------------------------------------------------------- 7. run from another cwd with relative paths
    rc, out = run(['solution/predict.py', '--images', os.path.relpath(test_dir, os.path.dirname(SOL)), '--out', os.path.join(work, 'relcwd.csv')], cwd=os.path.dirname(SOL))
    check('predict.py from parent dir with relative paths', rc == 0)

    # ---------------------------------------------------------------- 8. physics sanity
    straight = simulate(np.tile([1, 0, 0, 0], (50, 1)))
    left = simulate(np.tile([1, 1, 0, 0], (50, 1)))
    right = simulate(np.tile([1, 0, 0, 1], (50, 1)))
    stop = simulate(np.vstack([np.tile([1, 0, 0, 0], (20, 1)), np.tile([0, 0, 1, 0], (20, 1))]))
    check('physics: W drives straight up', abs(straight[-1, 0]) < 1e-6 and straight[-1, 1] < -100)
    check('physics: A turns left, D turns right', left[-1, 0] < -10 and right[-1, 0] > 10)
    check('physics: S brakes then reverses', stop[20, 3] > 0 and stop[-1, 3] < 0)

    print(f'\n{len(FAILS)} failures: {FAILS}')
    sys.exit(len(FAILS))


if __name__ == '__main__':
    main()
