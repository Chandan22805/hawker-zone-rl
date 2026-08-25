# Hawker Zone Placement with Deep Reinforcement Learning

This project trains a Deep Q-Network (DQN) to recommend locations for street-vendor stalls in Mumbai's G/North ward.

The area is divided into a grid. Each cell has:

- an estimated footfall score, based on nearby shops and amenities;
- a legal-placement flag; and
- an occupied/unoccupied state.

The DQN learns to place a fixed number of stalls in legal, high-footfall locations while avoiding duplicate placements and excessive spreading.

---

## How it works

```text
Mumbai ward map
      ↓
Footfall and legal-placement grid
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

This downloads OpenStreetMap shops and amenities, creates the grid, and saves `docs/g_north_data.npz`.

```bash
python src/build_footfall_grid.py
```

### 2. Train the DQN

This creates a new DQN with randomly initialized weights, trains it for 50,000 timesteps, and saves `hawker_dqn_gnorth.zip`.

```bash
python src/train_cnn.py
```

### 3. Run the greedy baseline

This provides a simple comparison by repeatedly selecting the currently highest-footfall legal cell.

```bash
python src/baseline_greedy.py
```

### 4. Visualize the trained placements

This loads the trained model and creates a map showing the recommended stalls:

```bash
python src/visualize_real.py
```

The output is saved as `real_map_demo.png`.

## Main files

- `src/hawker_env.py` — custom reinforcement-learning environment.
- `src/build_footfall_grid.py` — creates the footfall and legal-placement grid.
- `src/train_cnn.py` — trains the DQN from scratch.
- `src/baseline_greedy.py` — simple non-learning baseline.
- `src/visualize_real.py` — displays model placements on the real map.
- `docs/mumbai_wards.geojson` — Mumbai ward boundaries.
- `docs/g_north_data.npz` — generated model input data.

---

## Important note

The footfall value is an estimate based on mapped shops and amenities. It is not a direct measurement of pedestrian traffic.
