import numpy as np
from hawker_env import HawkerZoneEnv

env = HawkerZoneEnv(n_stalls=25)
env.load_real_footfall("docs/g_north_data.npz")
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