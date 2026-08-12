import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from hawker_env import HawkerZoneEnv

env = HawkerZoneEnv(n_stalls=15)
env.load_real_data("g_north_footfall_grid.npy")
model = DQN.load("hawker_dqn_cnn")

obs, info = env.reset()

for step in range(env.n_stalls):
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    q_values = model.q_net(obs_tensor).detach().numpy()[0]
    q_values[env.occupied.flatten() == 1] = -np.inf
    action = int(np.argmax(q_values))
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step={step}, action={action}, reward={reward:.2f}, placed_so_far={len(env.placed_cells)}")
    if terminated or truncated:
        break

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

im0 = axes[0].imshow(env.footfall, cmap="YlOrRd")
axes[0].set_title("Before: Footfall Density (OSM POIs)")
plt.colorbar(im0, ax=axes[0], label="Normalized footfall")

im1 = axes[1].imshow(env.footfall, cmap="YlOrRd")
rows = [i for i, j in env.placed_cells]
cols = [j for i, j in env.placed_cells]
axes[1].scatter(cols, rows, c="green", s=30, marker="s", label="Vendor stall")
axes[1].set_title("After: Vendor Placement over Footfall")
axes[1].legend()
plt.colorbar(im1, ax=axes[1], label="Normalized footfall")

plt.tight_layout()
plt.savefig("real_data_demo.png", bbox_inches="tight", dpi=150)
plt.show()