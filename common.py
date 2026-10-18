"""Shared pieces: image loading, the feature extractor, and the two-head WASD decoder.

The controls are split into two mutually exclusive axes, which is how they appear in the
clean Vice City data (W and S are never pressed together, nor A and D):
    longitudinal: 0 = none, 1 = W (accelerate), 2 = S (brake / reverse)
    lateral:      0 = none, 1 = A (left),       2 = D (right)
"""
import os
import re

import numpy as np
import torch
import torchvision
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
KEYS = ['W', 'A', 'S', 'D']
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Views given as fractions of the frame so any resolution works. The car crop is
# horizontally symmetric so a mirror flip maps it onto itself.
FULL_SIZE = (384, 224)            # (w, h) fed to the backbone
CAR_BOX = (0.25, 0.45, 0.75, 0.95)  # (x0, y0, x1, y1)
CAR_SIZE = (320, 192)


def natural_key(path):
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v')
VIDEO_FPS = 5  # frames sampled per second of video


def list_images(path):
    """Accept a folder, a single image, a video file (sampled with ffmpeg), or a CSV with a `filename` column."""
    if path.lower().endswith(VIDEO_EXTS) and os.path.isfile(path):
        import shutil
        import subprocess
        import tempfile
        if not shutil.which('ffmpeg'):
            raise SystemExit('video input needs ffmpeg on PATH (or extract the frames to a folder first)')
        out = tempfile.mkdtemp(prefix='route_frames_')
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', path, '-vf', f'fps={VIDEO_FPS}',
                        os.path.join(out, '%06d.png')], check=True)
        print(f'extracted frames from video at {VIDEO_FPS} fps into {out}')
        return list_images(out)
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(IMG_EXTS)]
        return sorted(files, key=natural_key)
    if path.lower().endswith('.csv'):
        import pandas as pd
        names = pd.read_csv(path)['filename'].astype(str).tolist()
        base = os.path.dirname(os.path.abspath(path))
        out = []
        for n in names:
            for cand in (n, os.path.join(base, n), os.path.join(base, 'test_images', n), os.path.join(base, 'images', n)):
                if os.path.exists(cand):
                    out.append(cand)
                    break
            else:
                out.append(os.path.join(base, n))  # missing; predictor falls back gracefully
        return out
    return [path]


def load_rgb(path):
    try:
        with Image.open(path) as im:
            return im.convert('RGB')
    except Exception as e:  # unreadable / corrupt
        print(f'[warn] could not read {path}: {e}')
        return None


def views(img):
    """Return the 2 view tensors (full frame, car crop), each 3xHxW in [0,1]."""
    w, h = img.size
    full = img.resize(FULL_SIZE, Image.BILINEAR)
    x0, y0, x1, y1 = CAR_BOX
    car = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))).resize(CAR_SIZE, Image.BILINEAR)
    to_t = lambda im: torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
    return to_t(full), to_t(car)


def build_backbone(name='convnext_tiny', pretrained=True):
    if name == 'convnext_tiny':
        m = torchvision.models.convnext_tiny(weights='DEFAULT' if pretrained else None)
        m.classifier[2] = torch.nn.Identity()  # keep LayerNorm + flatten, drop the 1000-way head
    elif name == 'efficientnet_b0':
        m = torchvision.models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        m.classifier = torch.nn.Identity()
    elif name == 'resnet50':
        m = torchvision.models.resnet50(weights='DEFAULT' if pretrained else None)
        m.fc = torch.nn.Identity()
    else:
        raise ValueError(name)
    return m.eval()


@torch.no_grad()
def extract(backbone, paths, batch=8, flip=False, log_every=0):
    """Features per image: concat(full-frame embedding, car-crop embedding). Returns (N, D), ok-mask."""
    feats, ok = [], []
    for i in range(0, len(paths), batch):
        fulls, cars = [], []
        for p in paths[i:i + batch]:
            img = load_rgb(p)
            ok.append(img is not None)
            if img is None:
                img = Image.new('RGB', (1366, 768))
            f, c = views(img)
            if flip:
                f, c = f.flip(-1), c.flip(-1)
            fulls.append(f)
            cars.append(c)
        f = backbone((torch.stack(fulls) - MEAN) / STD)
        c = backbone((torch.stack(cars) - MEAN) / STD)
        feats.append(torch.cat([f, c], 1).numpy())
        if log_every and (i // batch) % log_every == 0:
            print(f'  features {min(i + batch, len(paths))}/{len(paths)}', flush=True)
    return np.concatenate(feats), np.array(ok)


def labels_to_axes(wasd):
    """(N,4) WASD -> (long, lat) class ids."""
    wasd = np.asarray(wasd)
    lon = np.where(wasd[:, 0] == 1, 1, np.where(wasd[:, 2] == 1, 2, 0))
    lat = np.where(wasd[:, 1] == 1, 1, np.where(wasd[:, 3] == 1, 2, 0))
    return lon, lat


def axes_to_wasd(lon, lat):
    lon, lat = np.asarray(lon), np.asarray(lat)
    return np.stack([(lon == 1), (lat == 1), (lon == 2), (lat == 2)], 1).astype(int)


def swap_lat_probs(p):
    """Mirror flip swaps left/right: reorder lateral probs [none, A, D] -> [none, D, A]."""
    return p[:, [0, 2, 1]]


class WASDModel:
    """Frozen backbone + two multinomial logistic-regression heads (sklearn) with per-class log-bias."""

    def __init__(self, backbone_name, backbone, scaler, head_lon, head_lat, bias_lon, bias_lat):
        self.backbone_name, self.backbone = backbone_name, backbone
        self.scaler, self.head_lon, self.head_lat = scaler, head_lon, head_lat
        self.bias_lon, self.bias_lat = np.asarray(bias_lon), np.asarray(bias_lat)

    def probs_from_feats(self, f, f_flip=None):
        x = self.scaler.transform(f)
        p_lon, p_lat = self.head_lon.predict_proba(x), self.head_lat.predict_proba(x)
        if f_flip is not None:  # test-time mirror augmentation
            xf = self.scaler.transform(f_flip)
            p_lon = (p_lon + self.head_lon.predict_proba(xf)) / 2
            p_lat = (p_lat + swap_lat_probs(self.head_lat.predict_proba(xf))) / 2
        return p_lon, p_lat

    def decide(self, p_lon, p_lat):
        lon = np.argmax(np.log(p_lon + 1e-9) + self.bias_lon, 1)
        lat = np.argmax(np.log(p_lat + 1e-9) + self.bias_lat, 1)
        return axes_to_wasd(lon, lat)

    def predict_paths(self, paths, tta=True, log_every=0):
        f, ok = extract(self.backbone, paths, log_every=log_every)
        ff = extract(self.backbone, paths, flip=True)[0] if tta else None
        p_lon, p_lat = self.probs_from_feats(f, ff)
        wasd = self.decide(p_lon, p_lat)
        wasd[~ok] = [1, 0, 0, 0]  # unreadable frame: keep driving straight
        return wasd, p_lon, p_lat, ok

    def save(self, path):
        """Heads are stored as plain arrays so inference needs only torch + numpy (no sklearn pickle)."""
        def head(h):
            assert list(h.classes_) == [0, 1, 2], h.classes_
            return {'coef': torch.tensor(h.coef_), 'intercept': torch.tensor(h.intercept_)}
        half = {k: v.half() if v.is_floating_point() else v for k, v in self.backbone.state_dict().items()}
        torch.save({'backbone_name': self.backbone_name, 'backbone_state': half,
                    'scaler': {'mean': torch.tensor(self.scaler.mean_), 'scale': torch.tensor(self.scaler.scale_)},
                    'head_lon': head(self.head_lon), 'head_lat': head(self.head_lat),
                    'bias_lon': self.bias_lon.tolist(), 'bias_lat': self.bias_lat.tolist()}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location='cpu', weights_only=True)
        bb = build_backbone(ck['backbone_name'], pretrained=False)
        bb.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck['backbone_state'].items()})
        return cls(ck['backbone_name'], bb.eval(), _Scaler(**ck['scaler']), _Softmax(**ck['head_lon']),
                   _Softmax(**ck['head_lat']), ck['bias_lon'], ck['bias_lat'])


class _Scaler:
    def __init__(self, mean, scale):
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)

    def transform(self, x):
        return (x - self.mean) / self.scale


class _Softmax:
    """Multinomial logistic regression inference: softmax(x @ coef.T + intercept)."""

    def __init__(self, coef, intercept):
        self.coef, self.intercept = np.asarray(coef), np.asarray(intercept)

    def predict_proba(self, x):
        z = x @ self.coef.T + self.intercept
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)
