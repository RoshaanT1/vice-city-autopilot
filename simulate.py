"""Vice City Autopilot - https://github.com/RoshaanT1/vice-city-autopilot

Pygame self-driving simulation: the model reads each GTA Vice City frame of a route, predicts W/A/S/D,
and those controls drive a car on a top-down map, tracing the path it takes.

If ground-truth controls are available (a --labels CSV, or filenames ending in _WASD like 85638_0011.png),
a ghost car drives the true route alongside so you can see whether the model follows it.

usage:
  python simulate.py --images <folder of frames | csv with filename column> [--labels truth.csv]
  python simulate.py --images ../DATA-QUEST-R02-SDC-GTA-VICE-CITY-master/test_images
  python simulate.py --images frames/ --record simulation.mp4 --headless   # render a video, no window
  python simulate.py --images frames/ --headless                          # no window; writes outputs only

keys: SPACE pause | LEFT/RIGHT step (paused) | UP/DOWN speed | R restart | ESC/Q quit
outputs (in --out, default sim_output/): predictions.csv, trajectory.csv, path.png, summary.json
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from common import KEYS, WASDModel, list_images, natural_key

HERE = os.path.dirname(os.path.abspath(__file__))
W_WIN, H_WIN = 1280, 720
CAM_W, CAM_H = 640, 360
MAP_X, MAP_W, MAP_H = 650, 620, 700

# car dynamics, units per frame
ACCEL, BRAKE, REVERSE, FRICTION = 0.55, 1.1, 0.35, 0.06
MAX_V, MAX_REV = 6.0, -2.5
TURN_DEG = 5.0

COL = dict(bg=(18, 18, 28), panel=(30, 30, 46), grid=(42, 42, 62), text=(235, 235, 245), dim=(150, 150, 170),
           pred=(255, 60, 170), truth=(60, 220, 255), key_on=(255, 60, 170), key_off=(60, 60, 80),
           ok=(80, 220, 120), bad=(240, 80, 80))


def simulate(wasd):
    """Kinematic car driven by per-frame WASD. Returns (N+1, 4) array of x, y, heading(rad), speed."""
    x = y = v = 0.0
    th = -math.pi / 2  # start facing up the map
    out = [(x, y, th, v)]
    for w, a, s, d in wasd:
        if w:
            v = min(MAX_V, v + ACCEL)
        elif s:
            v = v - BRAKE if v > 0 else max(MAX_REV, v - REVERSE)
            v = max(v, MAX_REV)
        else:
            v = math.copysign(max(0.0, abs(v) - FRICTION * max(1, abs(v))), v)
        steer = (d - a) * math.radians(TURN_DEG) * min(1.0, abs(v) / 2.0)
        th += steer if v >= 0 else -steer  # steering inverts in reverse, like a real car
        x += v * math.cos(th)
        y += v * math.sin(th)
        out.append((x, y, th, v))
    return np.array(out)


def smooth_probs(p, k):
    """Centred moving average over k frames (k made odd); edges padded by repetition."""
    if k <= 1:
        return p
    k += 1 - k % 2
    pad = k // 2
    pp = np.pad(p, ((pad, pad), (0, 0)), mode='edge')
    ker = np.ones(k) / k
    return np.stack([np.convolve(pp[:, j], ker, mode='valid') for j in range(p.shape[1])], 1)


def truth_from_names(paths):
    rows = []
    for p in paths:
        m = re.search(r'_([01]{4})\.[A-Za-z]+$', os.path.basename(p))
        if not m:
            return None
        rows.append([int(c) for c in m.group(1)])
    return np.array(rows)


def load_truth(labels_csv, paths):
    if labels_csv:
        df = pd.read_csv(labels_csv)
        df['filename'] = df['filename'].astype(str)
        lut = {r.filename: [int(r[k]) for k in KEYS] for _, r in df.iterrows()}
        names = [os.path.basename(p) for p in paths]
        if all(n in lut for n in names):
            return np.array([lut[n] for n in names])
        print(f'[warn] --labels does not cover all frames ({sum(n in lut for n in names)}/{len(names)}); ignoring')
        return None
    return truth_from_names(paths)


def compare(pred, truth, traj_p, traj_t):
    dev = np.linalg.norm(traj_p[:, :2] - traj_t[:, :2], axis=1)
    path_len = float(np.sum(np.linalg.norm(np.diff(traj_t[:, :2], axis=0), axis=1))) or 1.0
    return {'frames': int(len(pred)), 'exact_match': float((pred == truth).all(1).mean()),
            'key_accuracy': {k: float((pred[:, i] == truth[:, i]).mean()) for i, k in enumerate(KEYS)},
            'mean_key_accuracy': float((pred == truth).mean()),
            'mean_path_deviation': float(dev.mean()), 'final_position_error': float(dev[-1]),
            'true_path_length': path_len, 'final_error_pct_of_path': float(100 * dev[-1] / path_len)}


MAP_MARGIN, MAP_PAD, MIN_SPAN = 50, 18, 80.0


class MapView:
    """World -> screen transform for the map panel.

    update() auto-fits everything drawn so far: it starts zoomed in on the car and zooms out / re-centres as the
    route grows, eased so the camera glides. If easing would ever let a point come within MAP_PAD px of the panel
    edge, it snaps to the fitted view, so the path can never leave the panel."""

    def __init__(self, pts=None):
        self.scale = self.center = None
        self.snaps = 0
        if pts is not None:
            self.scale, self.center = self.target(pts)

    @staticmethod
    def target(pts):
        lo, hi = pts.min(0), pts.max(0)
        span = np.maximum(hi - lo, MIN_SPAN)
        scale = min((MAP_W - 2 * MAP_MARGIN) / span[0], (MAP_H - 2 * MAP_MARGIN) / span[1], 6.0)
        return scale, (lo + hi) / 2

    def many(self, pts):
        sx = MAP_X + MAP_W / 2 + (pts[:, 0] - self.center[0]) * self.scale
        sy = 10 + MAP_H / 2 + (pts[:, 1] - self.center[1]) * self.scale
        return np.stack([sx, sy], 1)

    def inside(self, pts, pad=MAP_PAD):
        s = self.many(pts)
        return bool((s[:, 0] >= MAP_X + pad).all() and (s[:, 0] <= MAP_X + MAP_W - pad).all()
                    and (s[:, 1] >= 10 + pad).all() and (s[:, 1] <= 10 + MAP_H - pad).all())

    def update(self, pts, ease=0.12):
        ts, tc = self.target(pts)
        if self.scale is None:
            self.scale, self.center = ts, tc
            return
        self.scale += (ts - self.scale) * ease
        self.center = self.center + (tc - self.center) * ease
        if not self.inside(pts):
            self.scale, self.center = ts, tc
            self.snaps += 1

    def __call__(self, x, y):
        return (MAP_X + MAP_W / 2 + (x - self.center[0]) * self.scale,
                10 + MAP_H / 2 + (y - self.center[1]) * self.scale)


def car_poly(view, x, y, th, size=11):
    pts = [(1.4, 0), (-1, 0.7), (-0.6, 0), (-1, -0.7)]
    c, s = math.cos(th), math.sin(th)
    sx, sy = view(x, y)
    return [(sx + size * (px * c - py * s), sy + size * (px * s + py * c)) for px, py in pts]


def interp_pose(traj, t):
    """Pose after t frames of driving (t may be fractional): linear in position/speed, shortest-arc in heading."""
    i = min(int(t), len(traj) - 2)
    u = min(max(t - i, 0.0), 1.0)
    a, b = traj[i], traj[i + 1]
    dth = (b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi
    return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + dth * u, a[3] + (b[3] - a[3]) * u


def run_window(paths, pred, probs, truth, traj_p, traj_t, out_dir, fps, autoclose, record=None, render_fps=60):
    """fps = route frames per second. The screen renders at render_fps; the car glides between frames and the
    camera crossfades into the next frame, so playback looks like video rather than a slideshow."""
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((W_WIN, H_WIN))
    pygame.display.set_caption('GTA Vice City - self-driving simulation')
    font, big, small = pygame.font.Font(None, 26), pygame.font.Font(None, 44), pygame.font.Font(None, 20)
    view = MapView()
    clock = pygame.time.Clock()
    n = len(paths)
    t, paused, running, saved_end = 0.0, False, True, False
    cache = {}
    enc = None
    if record:
        autoclose, render_fps = True, 30
        enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                '-s', f'{W_WIN}x{H_WIN}', '-r', str(render_fps), '-i', '-', '-pix_fmt', 'yuv420p', record],
                               stdin=subprocess.PIPE)
    to_bytes = getattr(pygame.image, 'tobytes', None) or pygame.image.tostring

    # Decoding + scaling a 1366x768 PNG takes ~20 ms, over the 16.7 ms budget of a 60 fps frame: a hitch on every
    # frame change.
    # A background thread decodes + scales the next frames ahead of playback; the loop only uploads bytes.
    import threading
    from PIL import Image
    ready, lock, stop, want = {}, threading.Lock(), threading.Event(), {'i': 0}

    def decode(k):
        try:
            with Image.open(paths[k]) as im:
                return im.convert('RGB').resize((CAM_W, CAM_H), Image.BILINEAR).tobytes()
        except Exception:
            return bytes(CAM_W * CAM_H * 3)

    def prefetch():
        while not stop.is_set():
            i0 = want['i']
            todo = [k for k in range(i0, min(n, i0 + 16)) if k not in ready]
            if not todo:
                stop.wait(0.005)
                continue
            b = decode(todo[0])
            with lock:
                ready[todo[0]] = b
                for old in [j for j in ready if j < i0 - 2 or j > i0 + 24]:
                    ready.pop(old, None)

    threading.Thread(target=prefetch, daemon=True).start()

    def frame_surface(k):
        if k not in cache:
            with lock:
                b = ready.get(k)
            if b is None:  # jumped ahead with the arrow keys: decode now
                b = decode(k)
            cache[k] = pygame.image.frombuffer(b, (CAM_W, CAM_H), 'RGB').convert()
            if len(cache) > 30:
                cache.pop(next(iter(cache)))
        return cache[k]

    def draw_keys(x0, y0, keys, ref):
        layout = {'W': (1, 0), 'A': (0, 1), 'S': (1, 1), 'D': (2, 1)}
        for j, k in enumerate(KEYS):
            cx, cy = layout[k]
            r = pygame.Rect(x0 + cx * 62, y0 + cy * 62, 56, 56)
            pygame.draw.rect(screen, COL['key_on'] if keys[j] else COL['key_off'], r, border_radius=8)
            if ref is not None:
                pygame.draw.rect(screen, COL['ok'] if keys[j] == ref[j] else COL['bad'], r, 3, border_radius=8)
            txt = big.render(k, True, COL['text'])
            screen.blit(txt, txt.get_rect(center=r.center))

    def drawn(traj, t):
        """Points of the path driven up to time t (ending at the interpolated car position) and the car pose."""
        i = min(int(t), len(traj) - 2)
        x, y, th, _ = interp_pose(traj, t)
        return np.vstack([traj[:i + 1, :2], [[x, y]]]), (x, y, th)

    def draw_path(traj, t, color, width):
        pts, (x, y, th) = drawn(traj, t)
        if len(pts) >= 2:
            pygame.draw.lines(screen, color, False, view.many(pts).tolist(), width)
        pygame.draw.polygon(screen, color, car_poly(view, x, y, th))

    def draw_grid():
        """World-aligned grid (~60 px spacing at any zoom) so the map visibly moves and zooms with the route."""
        raw = 60 / view.scale
        mag = 10 ** math.floor(math.log10(raw))
        step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
        half_w, half_h = MAP_W / 2 / view.scale, MAP_H / 2 / view.scale
        gx = math.ceil((view.center[0] - half_w) / step) * step
        while gx <= view.center[0] + half_w:
            sx = view(gx, 0)[0]
            pygame.draw.line(screen, COL['grid'], (sx, 10), (sx, 10 + MAP_H))
            gx += step
        gy = math.ceil((view.center[1] - half_h) / step) * step
        while gy <= view.center[1] + half_h:
            sy = view(0, gy)[1]
            pygame.draw.line(screen, COL['grid'], (MAP_X, sy), (MAP_X + MAP_W, sy))
            gy += step

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q)):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE: paused = not paused
                elif e.key == pygame.K_RIGHT: t = min(float(int(t) + 1), n - 1.0)
                elif e.key == pygame.K_LEFT: t = max(float(int(t) - 1), 0.0)
                elif e.key == pygame.K_UP: fps = min(60, fps + 2)
                elif e.key == pygame.K_DOWN: fps = max(1, fps - 2)
                elif e.key == pygame.K_r: t, saved_end = 0.0, False
        if not running:
            break
        i = min(int(t), n - 1)
        want['i'] = i
        u = t - i if t < n else 1.0
        screen.fill(COL['bg'])
        # camera panel: hold the frame, then a quick crossfade into the next one in the last 20% of its slot
        screen.blit(frame_surface(i), (5, 5))
        fade = min(max((u - 0.8) / 0.2, 0.0), 1.0)
        if fade > 0 and i + 1 < n:
            nxt = frame_surface(i + 1).copy()
            nxt.set_alpha(int(255 * fade))
            screen.blit(nxt, (5, 5))
        pygame.draw.rect(screen, COL['panel'], (5, CAM_H + 10, CAM_W, H_WIN - CAM_H - 15), border_radius=6)
        draw_keys(20, CAM_H + 25, pred[i], truth[i] if truth is not None else None)
        p_lon, p_lat = probs
        speed = interp_pose(traj_p, min(t, n))[3]
        lines = [f'frame {i + 1}/{n}  {os.path.basename(paths[i])}',
                 f'model: {"".join(k for k, v in zip(KEYS, pred[i]) if v) or "-"}   '
                 f'p W {p_lon[i, 1]:.2f}  A {p_lat[i, 1]:.2f}  S {p_lon[i, 2]:.2f}  D {p_lat[i, 2]:.2f}',
                 f'speed {speed:.1f}   {fps} frames/s{"  [PAUSED]" if paused else ""}']
        if truth is not None:
            seen = slice(0, i + 1)
            acc = (pred[seen] == truth[seen]).mean()
            dev = math.dist(interp_pose(traj_p, min(t, n))[:2], interp_pose(traj_t, min(t, n))[:2])
            lines.append(f'truth: {"".join(k for k, v in zip(KEYS, truth[i]) if v) or "-"}   '
                         f'running key-acc {acc:.1%}   path deviation {dev:.1f}')
        for j, ln in enumerate(lines):
            screen.blit(font.render(ln, True, COL['text'] if j != 2 else COL['dim']), (220, CAM_H + 25 + j * 30))
        screen.blit(small.render('SPACE pause  <-/-> step  UP/DOWN speed  R restart  ESC quit', True, COL['dim']), (20, H_WIN - 25))
        # map panel: auto-fit camera that zooms out as the route grows, so the path never leaves the panel
        tt = min(t, n)
        trajs = [traj_p] + ([traj_t] if traj_t is not None else [])
        fit_pts = []
        for tr_ in trajs:  # path so far + a look-ahead point, so there is always room in front of the car
            pts_, (cx, cy, cth) = drawn(tr_, tt)
            fit_pts += [pts_, [[cx + 30 * math.cos(cth), cy + 30 * math.sin(cth)]]]
        view.update(np.vstack(fit_pts))
        map_rect = pygame.Rect(MAP_X, 10, MAP_W, MAP_H)
        pygame.draw.rect(screen, COL['panel'], map_rect, border_radius=6)
        screen.set_clip(map_rect)  # last-resort safety: nothing can ever be drawn outside the map panel
        draw_grid()
        if traj_t is not None:
            draw_path(traj_t, tt, COL['truth'], 3)
        draw_path(traj_p, tt, COL['pred'], 3)
        screen.set_clip(None)
        legend = [('model path', COL['pred'])] + ([('true path', COL['truth'])] if traj_t is not None else [])
        for j, (name, c) in enumerate(legend):
            pygame.draw.rect(screen, c, (MAP_X + 15, 25 + j * 24, 14, 14))
            screen.blit(font.render(name, True, COL['text']), (MAP_X + 36, 23 + j * 24))
        pygame.display.flip()
        if enc:
            enc.stdin.write(to_bytes(screen, 'RGB'))
        if t >= n:
            if not saved_end:
                pygame.image.save(screen, os.path.join(out_dir, 'path.png'))
                saved_end = True
            if autoclose:
                running = False
            paused = True
        if not paused:
            t = min(t + fps / render_fps, float(n))
        clock.tick(render_fps if not enc else 0)
    stop.set()
    pygame.quit()
    if enc:
        enc.stdin.close()
        enc.wait()
        print(f'recorded {record}' if enc.returncode == 0 else f'[warn] ffmpeg exited with {enc.returncode}')


def save_path_png(traj_p, traj_t, out_dir):
    """Headless rendering of the final map."""
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    import pygame
    pygame.init()
    surf = pygame.Surface((W_WIN, H_WIN))
    surf.fill(COL['bg'])
    view = MapView(np.vstack([traj_p[:, :2]] + ([traj_t[:, :2]] if traj_t is not None else [])))
    for traj, c in ([(traj_t, COL['truth'])] if traj_t is not None else []) + [(traj_p, COL['pred'])]:
        pygame.draw.lines(surf, c, False, [view(*p[:2]) for p in traj], 3)
        pygame.draw.polygon(surf, c, car_poly(view, *traj[-1, :3]))
    pygame.image.save(surf, os.path.join(out_dir, 'path.png'))
    pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', required=True, help='route frames: folder, single image, or CSV with filename column')
    ap.add_argument('--labels', help='optional ground-truth CSV (filename,W,A,S,D) to compare paths')
    ap.add_argument('--model', default=os.path.join(HERE, 'model.pt'))
    ap.add_argument('--out', default='sim_output')
    ap.add_argument('--fps', type=int, default=8, help='route frames per second')
    ap.add_argument('--smooth', type=int, default=3,
                    help='average model probabilities over this many neighbouring frames to remove steering flicker (1 = off)')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--autoclose', action='store_true', help='close the window when the route ends')
    ap.add_argument('--record', help='save the simulation as a video, e.g. --record simulation.mp4 (needs ffmpeg)')
    ap.add_argument('--keep-order', action='store_true', help='drive frames in the given CSV order instead of filename order')
    args = ap.parse_args()
    if args.record and args.headless:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')  # render off-screen but still run the full window loop
        args.headless = False

    if not os.path.isfile(args.model):
        sys.exit(f'model file not found: {args.model} (pass --model path/to/model.pt)')
    paths = list_images(args.images)
    if not paths:
        sys.exit(f'no images found at {args.images}')
    if not args.keep_order:  # a route is driven in time order: 2.png before 10.png, regardless of CSV row order
        paths = sorted(paths, key=natural_key)
    os.makedirs(args.out, exist_ok=True)
    print(f'{len(paths)} frames; loading model {args.model}')
    model = WASDModel.load(args.model)
    pred, p_lon, p_lat, ok = model.predict_paths(paths, log_every=5)
    if args.smooth > 1:
        if ok.any() and not ok.all():  # unreadable frames must not drag their neighbours
            p_lon[~ok], p_lat[~ok] = p_lon[ok].mean(0), p_lat[ok].mean(0)
        p_lon, p_lat = smooth_probs(p_lon, args.smooth), smooth_probs(p_lat, args.smooth)
        pred = model.decide(p_lon, p_lat)
        pred[~ok] = [1, 0, 0, 0]
    truth = load_truth(args.labels, paths)
    traj_p = simulate(pred)
    traj_t = simulate(truth) if truth is not None else None

    names = [os.path.basename(p) for p in paths]
    pd.DataFrame(pred, columns=KEYS).assign(filename=names)[['filename'] + KEYS].to_csv(os.path.join(args.out, 'predictions.csv'), index=False)
    tr = pd.DataFrame(traj_p[1:], columns=['x', 'y', 'heading', 'speed']).assign(filename=names)
    if traj_t is not None:
        tr[['true_x', 'true_y']] = traj_t[1:, :2]
    tr.to_csv(os.path.join(args.out, 'trajectory.csv'), index=False)
    summary = {'frames': len(paths), 'unreadable': int((~ok).sum()), 'smoothing_window': args.smooth,
               'predicted_key_rates': {k: float(pred[:, i].mean()) for i, k in enumerate(KEYS)}}
    if truth is not None:
        summary['vs_truth'] = compare(pred, truth, traj_p, traj_t)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

    if args.headless:
        save_path_png(traj_p, traj_t, args.out)
    else:
        run_window(paths, pred, (p_lon, p_lat), truth, traj_p, traj_t, args.out, args.fps, args.autoclose,
                   os.path.abspath(args.record) if args.record else None)
    print(f'outputs in {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
