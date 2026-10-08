import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from shapely.ops import unary_union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hawker_env import HawkerZoneEnv

WARD_NAMES = ["G/N", "F/N", "G/S", "F/S"]
CELL_SIZE_DEG = 0.00025
N_STALLS = 100
ward_tag = "_".join(w.replace("/", "") for w in WARD_NAMES).lower()
cell_size_label = format(CELL_SIZE_DEG, "g")

wards = gpd.read_file("docs/mumbai_wards.geojson")
selected_wards = wards[wards["name"].isin(WARD_NAMES)]
boundary_polygon = unary_union(selected_wards.geometry)

buildings = ox.features_from_polygon(boundary_polygon, tags={"building": True})
roads_all = ox.features_from_polygon(boundary_polygon, tags={"highway": True})
road_lines = roads_all[roads_all.geometry.type.isin(["LineString", "MultiLineString"])]
road_points = roads_all[roads_all.geometry.type == "Point"]

minx, miny, maxx, maxy = boundary_polygon.bounds

env = HawkerZoneEnv(n_stalls=N_STALLS, max_steps=N_STALLS)
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

cell_size_deg = (maxx - minx) / env.footfall.shape[1]
stall_lons = [minx + (j + 0.5) * cell_size_deg for i, j in env.placed_cells]
stall_lats = [miny + (i + 0.5) * cell_size_deg for i, j in env.placed_cells]

fig, ax = plt.subplots(figsize=(10, 10))
buildings.plot(ax=ax, color="#c9c9c9", edgecolor="none", zorder=1)
road_lines.plot(ax=ax, color="#555555", linewidth=0.6, zorder=2)
road_points.plot(ax=ax, color="#888888", markersize=3, zorder=2)
gpd.GeoSeries([boundary_polygon]).boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=3)
ax.scatter(stall_lons, stall_lats, c="orange", s=12, marker="s", label="Vendor stall (greedy)", zorder=4)
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_title(f"Greedy Baseline Placement | {len(env.placed_cells)} stalls")
ax.legend()
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.show()
