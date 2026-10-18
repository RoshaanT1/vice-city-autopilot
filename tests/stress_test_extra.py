"""Extra edge-case stress tests: exotic formats/sizes, odd paths, missing files, determinism, long routes,
scripted keyboard input in the real pygame loop.
usage: python tests/stress_test_extra.py --data <repo folder> --work <scratch dir>
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SOL)
PY = sys.executable
FAILS = []


def check(name, cond, info=''):
    print(f'[{"PASS" if cond else "FAIL"}] {name} {info}', flush=True)
    if not cond:
        FAILS.append(name)


def run(args, cwd, env=None, timeout=900):
    r = subprocess.run([PY] + args, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=dict(os.environ, **(env or {})))
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--work', required=True)
    a = ap.parse_args()
    data, work = os.path.abspath(a.data), os.path.abspath(a.work)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    test_dir = os.path.join(data, 'test_images')
    src = [os.path.join(test_dir, f) for f in sorted(os.listdir(test_dir))[:6]]
    base = Image.open(src[0]).convert('RGB')

    # 1. exotic formats and sizes, mixed in one route, inside a unicode + space directory name
    odd = os.path.join(work, 'route ünïcødé dir')
    os.makedirs(odd)
    base.convert('I;16').save(os.path.join(odd, '01_16bit.png'))
    base.convert('P').save(os.path.join(odd, '02_palette.png'))
    base.convert('LA').save(os.path.join(odd, '03_gray_alpha.png'))
    base.convert('CMYK').save(os.path.join(odd, '04_cmyk.jpg'))
    base.save(os.path.join(odd, '05_webp.webp'))
    base.save(os.path.join(odd, '06_bitmap.bmp'))
    base.resize((768, 1366)).save(os.path.join(odd, '07_portrait.png'))
    base.resize((64, 36)).save(os.path.join(odd, '08_tiny.png'))
    base.resize((3840, 2160)).save(os.path.join(odd, '09_4k.png'))
    Image.new('RGB', (1, 1), (255, 0, 0)).save(os.path.join(odd, '10_one_pixel.png'))
    Image.new('RGB', (1366, 768), (0, 0, 0)).save(os.path.join(odd, '11_black.png'))
    Image.new('RGB', (1366, 768), (255, 255, 255)).save(os.path.join(odd, '12_white.png'))
    rc, out = run([os.path.join(SOL, 'predict.py'), '--images', odd + '/', '--out', os.path.join(work, 'odd.csv')], cwd=work)
    ok = rc == 0 and len(pd.read_csv(os.path.join(work, 'odd.csv'))) == 12
    check('exotic formats/sizes (16-bit, palette, LA, CMYK, webp, bmp, portrait, 64x36, 4K, 1px, black, white) + unicode dir + trailing slash',
          ok, out.strip().splitlines()[-1] if rc == 0 else out[-500:])
    rc, out = run([os.path.join(SOL, 'simulate.py'), '--images', odd, '--headless', '--out', os.path.join(work, 'sim_odd')], cwd=work)
    check('simulate.py on exotic route', rc == 0, '' if rc == 0 else out[-500:])

    # 2. determinism
    rc1, _ = run([os.path.join(SOL, 'predict.py'), '--images', test_dir, '--out', os.path.join(work, 'd1.csv')], cwd=work)
    rc2, _ = run([os.path.join(SOL, 'predict.py'), '--images', test_dir, '--out', os.path.join(work, 'd2.csv')], cwd=work)
    check('deterministic: two runs give identical predictions', rc1 == rc2 == 0 and
          pd.read_csv(os.path.join(work, 'd1.csv')).equals(pd.read_csv(os.path.join(work, 'd2.csv'))))

    # 3. symlinked folder
    os.symlink(test_dir, os.path.join(work, 'linked_route'))
    rc, out = run([os.path.join(SOL, 'simulate.py'), '--images', os.path.join(work, 'linked_route'), '--headless', '--out', os.path.join(work, 'sim_link')], cwd=work)
    check('symlinked route folder', rc == 0)

    # 4. CSV naming files that do not exist + labels that only partially cover the route
    pd.DataFrame({'filename': ['100.png', 'missing_1.png', '102.png', 'missing_2.png']}).to_csv(os.path.join(work, 'partial.csv'), index=False)
    rc, out = run([os.path.join(SOL, 'predict.py'), '--images', test_dir, '--csv', os.path.join(work, 'partial.csv'), '--out', os.path.join(work, 'partial_pred.csv')], cwd=work)
    pp = pd.read_csv(os.path.join(work, 'partial_pred.csv')) if rc == 0 else None
    check('CSV with missing files: no crash, row kept, falls back to W', rc == 0 and len(pp) == 4 and
          (pp[pp.filename.str.startswith('missing')][['W', 'A', 'S', 'D']].values == [1, 0, 0, 0]).all())
    pd.DataFrame({'filename': ['100.png'], 'W': [1], 'A': [0], 'S': [0], 'D': [0]}).to_csv(os.path.join(work, 'lab_partial.csv'), index=False)
    rc, out = run([os.path.join(SOL, 'simulate.py'), '--images', os.path.join(work, 'linked_route'), '--labels', os.path.join(work, 'lab_partial.csv'),
                   '--headless', '--out', os.path.join(work, 'sim_lab_partial')], cwd=work)
    check('--labels covering only part of the route: warns and continues', rc == 0 and 'does not cover' in out)

    # 5. missing model file gives a clean message
    rc, out = run([os.path.join(SOL, 'simulate.py'), '--images', test_dir, '--model', os.path.join(work, 'nope.pt'), '--headless'], cwd=work)
    check('missing model file: clean error, no traceback', rc != 0 and 'Traceback' not in out and 'model' in out.lower(), out.strip()[-200:])

    # 6. long route (200 frames, repeated frames) timing
    long_dir = os.path.join(work, 'long_route')
    os.makedirs(long_dir)
    tests = sorted(os.listdir(test_dir))
    for i in range(200):
        os.symlink(os.path.join(test_dir, tests[i % len(tests)]), os.path.join(long_dir, f'{i}.png'))
    t = time.time()
    rc, out = run([os.path.join(SOL, 'simulate.py'), '--images', long_dir, '--headless', '--out', os.path.join(work, 'sim_long')], cwd=work)
    dt = time.time() - t
    check('200-frame route completes', rc == 0, f'{dt:.0f}s total, {dt / 200:.2f}s/frame')

    # 7. scripted keyboard input through the real pygame event loop
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    import pygame
    import simulate as S
    pred = np.tile([1, 0, 0, 0], (12, 1)); pred[3:6] = [1, 1, 0, 0]; pred[8:10] = [0, 0, 1, 0]
    probs = (np.tile([0.1, 0.8, 0.1], (12, 1)), np.tile([0.8, 0.1, 0.1], (12, 1)))
    truth = np.tile([1, 0, 0, 0], (12, 1))
    script = [[pygame.K_SPACE], [pygame.K_RIGHT], [pygame.K_RIGHT], [pygame.K_LEFT], [pygame.K_UP] * 5, [pygame.K_DOWN] * 3,
              [pygame.K_r], [pygame.K_SPACE], [], [], [], [], [], [pygame.K_LEFT], [pygame.K_q]]
    real_get = pygame.event.get
    calls = {'n': 0}

    def fake_get():
        real_get()
        k = calls['n']; calls['n'] += 1
        keys = script[k] if k < len(script) else [pygame.K_ESCAPE]
        return [pygame.event.Event(pygame.KEYDOWN, key=key) for key in keys]
    pygame.event.get = fake_get
    try:
        S.run_window(src[:6] * 2, pred, probs, truth, S.simulate(pred), S.simulate(truth), work, 30, False)
        ok, info = True, f'{calls["n"]} loop iterations'
    except Exception as e:
        ok, info = False, repr(e)
    pygame.event.get = real_get
    check('window loop: SPACE / arrows / UP / DOWN / R / Q handled without error', ok, info)

    print(f'\n{len(FAILS)} failures: {FAILS}')
    sys.exit(len(FAILS))


if __name__ == '__main__':
    main()
