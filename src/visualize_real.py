import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from hawker_env import HawkerZoneEnv

wards = gpd.read_file("docs/mumbai_wards.geojson")
g_north = wards[wards["name"] == "G/N"]
boundary_polygon = g_north.geometry.iloc[0]

buildings = ox.features_from_polygon(boundary_polygon, tags={"building": True})
roads_all = ox.features_from_polygon(boundary_polygon, tags={"highway": True})
road_lines = roads_all[roads_all.geometry.type.isin(["LineString", "MultiLineString"])]
road_points = roads_all[roads_all.geometry.type == "Point"]

minx, miny, maxx, maxy = boundary_polygon.bounds

env = HawkerZoneEnv(n_stalls=25)
env.load_real_footfall("docs/g_north_data.npz")
model = DQN.load("hawker_dqn_gnorth")
obs, info = env.reset()

for step in range(env.n_stalls):
    q_values = model.q_net(model.policy.obs_to_tensor(obs)[0]).detach().cpu().numpy()[0]
    q_values[(env.legal.flatten() == 0) | (env.occupied.flatten() == 1)] = -np.inf
    action = int(np.argmax(q_values))
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward)
    if terminated or truncated:
        break

print(f"Placed {len(env.placed_cells)} stalls")

fig, ax = plt.subplots(figsize=(10, 10))
buildings.plot(ax=ax, color="#c9c9c9", edgecolor="none", zorder=1)
road_lines.plot(ax=ax, color="#555555", linewidth=0.6, zorder=2)
road_points.plot(ax=ax, color="#888888", markersize=3, zorder=2)
gpd.GeoSeries([boundary_polygon]).boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=3)

cell_size_deg = (maxx - minx) / env.footfall.shape[1]
stall_lons = [minx + (j + 0.5) * cell_size_deg for i, j in env.placed_cells]
stall_lats = [miny + (i + 0.5) * cell_size_deg for i, j in env.placed_cells]
ax.scatter(stall_lons, stall_lats, c="green", s=12, marker="s", label="Vendor stall", zorder=4)

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_title("G North Ward: Vendor Placement on Real Map")
ax.legend()
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.savefig("real_map_demo.png", dpi=150)
plt.show()