# Sliding-Window Training

Trains **one** DQN across the whole of Mumbai by sliding a fixed **400×400-cell** window over the city and training on each window in turn. Every window has the same shape, so the same model can later be applied to any other city by sliding the same window over it.

At `0.00025°` cells (~28m), one window covers about **11km × 11km**, and all of Mumbai takes **12 windows** (4 rows × 3 columns).

---

## Running a single window

Window ids are `rRR_cCC`: row counts up from the south, column from the west. **`r00_c00` is the first window** (south Mumbai).

| Script | One window |
|---|---|
| `build_footfall_grid.py` | `--only r00_c00` |
| `train_cnn.py` | `--max-windows 1` (the next unfinished window; `r00_c00` on a fresh start) |
| `visualize_masks.py` | `--window r00_c00` |
| `visualize_real.py` | `--window r00_c00` |

## Flow

Run from the **project root**:

```bash
python sliding_window/src/build_footfall_grid.py   # 1. build every window's grid
python sliding_window/src/visualize_masks.py       # 2. check the stitched masks
python sliding_window/src/train_cnn.py             # 3. train across the windows
python sliding_window/src/visualize_real.py        # 4. place stalls in every window
python sliding_window/src/visualize_real.py --cluster
```

### 1. Build windows

```bash
python sliding_window/src/build_footfall_grid.py
python sliding_window/src/build_footfall_grid.py --only r01_c01   # just one window
python sliding_window/src/build_footfall_grid.py --force          # rebuild existing
```

- Each window is cut from **one city-wide grid**, so neighbouring windows line up cell-for-cell.
- Cells outside the city boundary (sea, neighbouring districts) are marked **illegal**, so a mostly-sea window still has the full 400×400 shape.
- OSM is queried **per window**, so memory stays bounded however big the city is.
- Footfall is smoothed on a padded area and then cropped, so shops just across a window edge still count.
- Footfall is normalised **per window** (each window's busiest cell = 1.0).
- **Resumable:** windows that are already built are skipped. The first full build takes about an hour (~6 min per window, mostly OSM download).
- Windows with fewer than `MIN_LEGAL_CELLS` legal cells are kept but marked not trainable.

Output: `footfall_grid_data/mumbai_0.00025_w400/window_rXX_cYY.npz` plus `windows.json`, which lists every window with its origin, cell counts and trainable flag.

### 2. Visualize masks

`--window r00_c00` limits it to chosen windows. Stitches every built window's masks into one city-wide image straight from the saved `.npz` files, with no OSM download. The window boxes are drawn with their ids.

Output: `visualizations/mumbai_0.00025_w400/masks_visualization.png` and `mask_stats.json`.

### 3. Train

```bash
python sliding_window/src/train_cnn.py                          # all remaining windows
python sliding_window/src/train_cnn.py --max-windows 2          # train 2, then stop
python sliding_window/src/train_cnn.py --timesteps-per-window 100000
python sliding_window/src/train_cnn.py --new-pass               # go over every window again
```

- Windows train in the order of `TRAIN_WINDOWS` in `config.py`: `r00_c00 → r01_c00 → r01_c01 → r02_c00 → r02_c01 → r03_c00 → r03_c01`, row by row from the south, west then east. The near-empty sea windows are left out.
- If the next window isn't built yet, it trains up to that point, then stops and prints the build command.
- The model is **always saved to the same file** after every window. A progress file records which windows are done.
- **Pause/resume:** stop at any point (or use `--max-windows`) and re-run. Training picks up from the saved model at the next unfinished window.
- Each window restarts the exploration schedule, so new areas get explored instead of inheriting the last window's low epsilon.

Output: `models/hawker_dqn_mumbai_0.00025_w400_stalls100.zip` and `..._progress.json`.

### 4. Visualize results

```bash
python sliding_window/src/visualize_real.py
python sliding_window/src/visualize_real.py --cluster --cluster-eps-meters 400 --cluster-min-samples 3
python sliding_window/src/visualize_real.py --stalls 50   # stalls per window
python sliding_window/src/visualize_real.py --window r00_c00   # only this window
```

Places `--stalls` stalls in **every** trainable window, so every part of the city gets its own nearby zone, then plots them all on one city map.

Output: `visualizations/mumbai_0.00025_w400/results/result.png` (or `result_cluster.png`) and the matching `_eval.png`.

---

## Changes from the main pipeline (and why)

| Change | Why |
|---|---|
| Fixed 400×400 window instead of a grid shaped to the wards | One model needs one fixed input/action shape to train on many areas and transfer to other cities |
| Out-of-city cells marked illegal | Keeps every window the same shape |
| Observations stored as `uint8` (0–255) | A float 400×400×3 observation is ~1.9MB, so the 2000-step replay buffer would need ~7.7GB RAM; `uint8` cuts it to ~1.9GB |
| CNN feature layer 256 (SB3 default 512) | Keeps the network about the size of the earlier 299×319 model (~76M parameters). Measured peak VRAM is **1.87GB** |
| `exploration_fraction=0.3` (SB3 default 0.1) | The main model stopped exploring too early, the leading suspect for its over-spread placements |
| Window origin and cell size saved inside each `.npz` | Stalls are mapped to exact coordinates instead of re-deriving the cell size |
| Settings in one `config.py` | Ward, cell size and window settings were copy-pasted across scripts before |

## Things to know

- **Model file is ~1.2GB** (two copies of the network plus optimizer state, all needed to resume properly). It's overwritten in place each window, so it doesn't pile up. `sliding_window/models/` is git-ignored.
- **Forgetting across windows:** training windows one after another can make the model drift away from earlier windows. The replay buffer mixes windows within one run, but restarts empty when you resume. Running more than one pass (`--new-pass`) helps.
