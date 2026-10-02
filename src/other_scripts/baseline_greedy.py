import numpy as np
from src.hawker_env import HawkerZoneEnv

WARD_NAMES = ["G/N", "F/N", "G/S", "F/S"]
CELL_SIZE_DEG = 0.00035
ward_tag = "_".join(w.replace("/", "") for w in WARD_NAMES).lower()
cell_size_label = format(CELL_SIZE_DEG, "g")

PLACE_STALLS = 100  # match PLACE_STALLS in visualize_real.py for a fair comparison

env = HawkerZoneEnv(n_stalls=PLACE_STALLS, max_steps=PLACE_STALLS)
env.load_real_footfall(f"footfall_grid_data/{ward_tag}_data_{cell_size_label}.npz")
obs, info = env.reset()

total_reward = 0
for step in range(env.n_stalls):
    mask = (env.legal.flatten() == 1) & (env.occupied.flatten() == 0)
    masked_footfall = env.footfall.flatten().copy()
    masked_footfall[~mask] = -np.inf
    action = int(np.argmax(masked_footfall))
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Greedy baseline — total reward: {total_reward:.2f}, stalls placed: {len(env.placed_cells)}")
print(f"Greedy placements: {env.placed_cells}")