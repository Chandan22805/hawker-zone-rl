import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from stable_baselines3 import DQN
from hawker_env import HawkerZoneEnv

colors = ["#e0e0e0", "#6e6e6e", "#2b2b2b", "#d9534f", "#5cb85c"]
cmap = ListedColormap(colors)
labels = ["Empty space", "Building", "Road", "No-hawking zone", "Vendor stall"]

MODEL_PATH = "hawker_dqn"

env = HawkerZoneEnv()
model = DQN.load(MODEL_PATH)
obs, info = env.reset()
before_grid = env.cell_type.copy()

for step in range(env.n_stalls):
    q_values = model.q_net(model.policy.obs_to_tensor(obs)[0]).detach().numpy()[0]
    mask = (env.legal.flatten() == 1) & (env.occupied.flatten() == 0)
    q_values[~mask] = -np.inf
    action = int(np.argmax(q_values))
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

after_grid = env.cell_type.copy()
for (i, j) in env.placed_cells:
    after_grid[i, j] = 4

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, grid, title in zip(axes, [before_grid, after_grid], ["Before: Ward Layout", "After: Vendor Placement"]):
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
legend_elements = [Patch(facecolor=colors[i], label=labels[i]) for i in range(5)]
fig.legend(handles=legend_elements, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig("before_after_demo.png", bbox_inches="tight", dpi=150)
plt.show()