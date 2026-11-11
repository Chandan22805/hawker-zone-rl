# Sliding Window: Vijayawada (inference only)

Runs the **Mumbai-trained** sliding-window model on Vijayawada, with **no training on Vijayawada at all**. This is the cross-city generalisation test: does a model that learned to place hawker zones in Mumbai produce sensible zones in a city it has never seen?

---

## Flow

Run from the **project root**, after training in `sliding_window/` is done:

```bash
python sliding_window_vijayawada/src/run_build_footfall_grid.py   # 1. build Vijayawada's windows
python sliding_window_vijayawada/src/run_visualize_masks.py       # 2. check the masks
python sliding_window_vijayawada/src/run_visualize_real.py        # 3. place stalls with the Mumbai model
python sliding_window_vijayawada/src/run_visualize_real.py --cluster
python sliding_window_vijayawada/src/run_visualize_real.py --window r00_c00   # zoom to one window
```

Each `run_*.py` takes the same flags as the matching script in `sliding_window/src/`.

## How it works

- **No copied code.** Each `run_*.py` runs the matching `sliding_window/src` script with this folder's `src/config.py` in place of Mumbai's, so a fix to the Mumbai pipeline applies here automatically.
- **City boundary** is Vijayawada's municipal boundary from OpenStreetMap (61.9 km², matching the Vijayawada Municipal Corporation's official ~61.88 km²). It's fetched on the first run and saved to `data/vijayawada_boundary.geojson`.
- **Same cell size and window shape as the model** (`0.00025°`, 400×400). These must match, or the network can't read the grid.
- **Model:** loaded directly from `sliding_window/models/`. Nothing is trained or saved here.

## Outputs

- `footfall_grid_data/vijayawada_0.00025_w400/`: one `.npz` per window, plus `windows.json`
- `visualizations/vijayawada_0.00025_w400/`: `masks_visualization.png`, plus `results/result*.png`

## Things to know

- **Vijayawada is 425 × 390 cells, 25 rows (~700m) taller than one window**, so it splits into two windows: `r00_c00` (almost the whole city) and `r01_c00` (a thin strip along the northern edge). Each window gets its own `--stalls` placements, so the strip gets as many stalls as the whole rest of the city. Use `--window r00_c00` to leave it out.
- **Same data limits as Mumbai.** Parks, forests and other non-built land aren't excluded from the legal mask, and footfall comes from mapped shops and amenities. OSM coverage in Vijayawada may also be thinner than in Mumbai, which would weaken the footfall signal.
