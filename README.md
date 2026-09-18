# Hawker Zone Placement with Deep Reinforcement Learning

This project trains a Deep Q-Network (DQN) to recommend locations for street-vendor stalls in Mumbai. It currently covers the combined **G/North + F/North** ward area.

The area is divided into a grid. Each cell has:

- an estimated footfall score, based on nearby shops and amenities;
- a legal-placement flag; and
- an occupied/unoccupied state.

The DQN learns to place a fixed number of stalls in legal, high-footfall locations while avoiding duplicate placements and excessive spreading.

---

## How it works

```text
Mumbai ward map(s)
      ↓
Footfall and legal-placement grid
      ↓
Mask visualization (sanity check)
      ↓
Gymnasium reinforcement-learning environment
      ↓
DQN training
      ↓
Recommended stall locations on a map
```

The model is trained from scratch. No pretrained model is used.

---

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run commands from the project root.

---

## Usage

### 1. Build the geographic data

This downloads OpenStreetMap shops and amenities, unions the selected wards, builds the footfall and legal-placement masks at `cell_size_deg = 0.00045` (~50m cells), and saves the result to `footfall_grid_data/`.

```bash
python src/build_footfall_grid.py
```

Output: `footfall_grid_data/gn_fn_data_0.00045.npz`

### 2. Visualize the masks

This renders each obstacle mask (buildings, roads, footways, water, railways) plus the final legal mask, both standalone and overlaid on the real map. Useful for sanity-checking the grid before training.

```bash
python src/visualize_masks.py
```

Output: `visualizations/gn_fn/cell_size_0.00045/masks_visualization.png` and `masks_overlay_map.png`

### 3. Train the DQN

This creates a new DQN with randomly initialized weights, trains it for 50,000 timesteps, and saves the model.

```bash
python src/train_cnn.py
```

Output: `models/hawker_dqn_gn_fn_0.00045_stalls<N>.zip`

### 4. Visualize the trained placements

This loads the trained model, runs inference, and plots the recommended stall locations on the real map, plus an evaluation image combining the placement map with the legal-mask overlay.

```bash
python src/visualize_real.py
```

Output: `visualizations/gn_fn/cell_size_0.00045/results/result.png` and `result_eval.png`

---

## Main files

- `src/hawker_env.py` — custom reinforcement-learning environment.
- `src/build_footfall_grid.py` — creates the footfall and legal-placement grid.
- `src/visualize_masks.py` — visualizes the individual obstacle masks and final legal mask.
- `src/train_cnn.py` — trains the DQN from scratch.
- `src/visualize_real.py` — displays model placements on the real map and saves evaluation images.
- `src/other_scripts/` — supplementary scripts not part of the main flow (greedy baseline, all-wards overview).
- `docs/mumbai_wards.geojson` — Mumbai ward boundaries.
- `footfall_grid_data/` — generated model input data (footfall + legal masks).
- `models/` — saved trained DQN models.
- `visualizations/` — mask visualizations, overlays, and placement results, organized by ward and cell size.

---

## Important note

The footfall value is an estimate based on mapped shops and amenities. It is not a direct measurement of pedestrian traffic.
