# GTA Vice City Self-Driving: WASD Controller + Pygame Simulation

## Quick start
```bash
pip install -r requirements.txt
# drive any route (folder of frames, single frame, or CSV with a `filename` column)
python simulate.py --images /path/to/route_frames
# with ground truth, a ghost car drives the true route for comparison
python simulate.py --images /path/to/route_frames --labels truth.csv
# Kaggle-style submission
python predict.py --images ../test_images --csv ../test.csv --out submission.csv
```
In the simulation window: **SPACE** pause, **←/→** step through frames while paused, **↑/↓** speed up or slow down, **R** restart, **ESC** quit.
Each run writes `sim_output/predictions.csv`, `trajectory.csv`, `path.png` and `summary.json`.

Other options:
- `--headless` runs with no window.
- `--record sim.mp4` saves the run as a 30 fps video.
- `--images` also accepts a video file, which is sampled at 5 fps with ffmpeg.
- Frames are driven in natural file-name order; `--keep-order` uses the CSV's row order instead.

The simulation renders at 60 fps. The car glides between frames and the camera crossfades between frames.
**Steering smoothing:** `--smooth 3` (the default) averages the model's probabilities over 3 neighbouring frames, which halves steering flicker on the test route (51 → 24 changes). `--smooth 1` turns it off.
`submission.csv` is always the raw per-frame model output.

## Data cleaning (`train.py`, see `cleaning_report.csv`)
| Issue found | Count | Action |
|---|---|---|
| CSV rows naming files that don't exist (the frames were saved under a different ID prefix, same `_WASD` suffix) | 8 | re-linked to the real file |
| `.jpg` junk images: Michael Jackson photos, a toy car, GTA V / San Andreas / GTA VI screenshots and thumbnails, all at random sizes; 36 of them labelled W+S+D, which is impossible | 50 | dropped |
| Labels with W+S or A+D pressed together | (all were among the JPGs) | dropped |
| Exact duplicate images (by MD5 hash) | 0 | none found |
| Hidden metadata or appended data in the image files | 0 | scanned all 560 images |

The result is **408 clean 1366×768 Vice City frames**. All 408 were also checked by eye on contact sheets: sedan, taxi, scooter and on-foot gameplay, no junk.

## Model
- **Two-axis decoder.** In the clean data, W and S are never pressed together, and neither are A and D. So the model predicts two 3-way classes, *longitudinal* {none, W, S} and *lateral* {none, A, D}. This guarantees physically valid controls.
- **Features.** A frozen ImageNet EfficientNet-B0 backbone produces embeddings from two views: the full frame, and a crop around the player's vehicle (brake lights, vehicle yaw, wheel angle). The crop is defined as fractions of the frame, so any resolution works.
- **Heads.** Multinomial logistic regression (C=0.1, balanced class weights) on standardised features, chosen by stratified 5-fold cross-validation.
- **Mirror augmentation.** A horizontally flipped frame is the same scene with left and right swapped, so each training image is also added flipped with A and D swapped. Predictions average the original and mirrored frame (test-time augmentation).

## Validation (out-of-fold, never on training frames)
Stratified 5-fold cross-validation on the 408 clean frames. Every prediction comes from a model that never saw that frame.
Path metrics drive the simulated car over routes of 30 consecutive frames, once with the predicted keys and once with the true keys, and compare the two paths.

| Metric | **Model** | Always W | Random keys (label priors) |
|---|---|---|---|
| Macro-F1 over W/A/S/D | **0.36** | 0.23 | – |
| Route end-point error, % of path length (median) | **12.8%** | 20.7% | 21.2% |
| Mean path deviation (sim units) | 13.3 | 15.1 | 11.5 |
| Exact match (all 4 keys right) | 0.49 | 0.59 | 0.39 |
| Mean per-key accuracy | 0.80 | 0.86 | 0.77 |

Per-key F1: W 0.87, A 0.29, S 0.08, D 0.19.
Single-frame key prediction has a low ceiling, because a still image shows little about which keys are held.
Always-W wins on raw accuracy because 82% of frames are W, but it never turns, so its paths drift.
Tuning per-class decision biases looked better in-sample but made held-out paths worse, so it is off by default (`--tune-bias`).
ConvNeXt-Tiny features were no better than EfficientNet-B0, and EfficientNet-B0 is about 5× faster, so it is the shipped model.

## Robustness (`tests/stress_test.py`, 31 checks)
- **Resolution:** decisions are stable across 1920×1080, 1280×720 and 800×450 frames and a 4:3 stretch (92% of frames identical, 98% of keys).
- **Quality:** JPEG q60, RGBA input and ±25% brightness pass.
- **Messy input:** corrupt, zero-byte and non-image files fall back to "W". Unicode or space-containing names and uppercase extensions work, and frames are naturally sorted (`2.png` before `10.png`).
- **Known weakness:** grayscale or heavily blurred frames change about half of the decisions. `tests/robustness_cv.py` measured this honestly, scoring degraded copies of held-out frames:
  - **Accuracy holds up:** macro-F1 is 0.36 clean, 0.36 blur, 0.35 JPEG q40 and 0.36 low-res.
  - **Grayscale is the real failure:** macro-F1 drops to 0.24, and A/D F1 falls to about 0.05, so the model stops steering. Its colour cues (red tail and brake lights) are gone.
  - **Flips are uncertain frames:** the decisions that change were already coin-flips (median confidence 0.50, against 0.87–0.91 for stable frames).
  - **Augmentation was rejected:** training on degraded copies raised clean macro-F1 to 0.38 but made clean route end-error worse (26% vs 12.8%), so the shipped model is trained on clean frames only.
- **More edge cases** (`tests/stress_test_extra.py`) all pass:
  - formats: 16-bit, palette and gray-alpha PNGs, CMYK JPEG, WebP, BMP
  - sizes: 1×1, 64×36, 4K, portrait, all-black and all-white frames, mixed in one route
  - paths: unicode and space-containing directories, symlinks
  - missing inputs: CSV rows naming missing files, labels covering only part of the route, a missing model file (clean error)
  - behaviour: determinism, a 200-frame route, scripted keyboard input in the live window
- **Speed:** about 0.2 s per frame on a laptop CPU with test-time augmentation.

## Files
`common.py` (shared model code), `train.py` (cleaning, cross-validation, fit), `predict.py` (CSV predictions), `simulate.py` (pygame simulation), `model.pt` (trained weights, 8 MB, EfficientNet-B0 stored in fp16; decisions identical to fp32), `tests/stress_test.py` (robustness suite), `tests/path_eval.py` (path-following evaluation).
