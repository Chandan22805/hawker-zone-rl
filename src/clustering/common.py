"""Shared pipeline for the clustering experiments.

Each algorithm folder supplies only a `cluster_fn(xy_meters) -> labels`; this
module runs the trained model, clusters its stall placements, and saves the
images into that algorithm's own folder.
"""

import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from shapely.geometry import MultiPoint
from shapely.ops import unary_union
from stable_baselines3 import DQN

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hawker_env import HawkerZoneEnv

WARD_NAMES = ["G/N", "F/N", "G/S", "F/S"]
CELL_SIZE_DEG = 0.00025
TRAINED_STALLS = 100
PLACE_STALLS = 100
# Groups smaller than this are shown as isolated stalls, not zones.
MIN_ZONE_STALLS = 3

ward_tag = "_".join(w.replace("/", "") for w in WARD_NAMES).lower()
cell_size_label = format(CELL_SIZE_DEG, "g")


def place_stalls():
    """Run the trained DQN and return the env plus stall lon/lat arrays."""
    env = HawkerZoneEnv(n_stalls=PLACE_STALLS, max_steps=PLACE_STALLS)
    env.load_real_footfall(f"footfall_grid_data/{ward_tag}_data_{cell_size_label}.npz")
    model = DQN.load(os.path.join("models", f"hawker_dqn_{ward_tag}_{cell_size_label}_stalls{TRAINED_STALLS}"))
    obs, _ = env.reset()
    for _ in range(env.n_stalls):
        q_values = model.q_net(model.policy.obs_to_tensor(obs)[0]).detach().cpu().numpy()[0]
        q_values[(env.legal.flatten() == 0) | (env.occupied.flatten() == 1)] = -np.inf
        obs, _, terminated, truncated, _ = env.step(int(np.argmax(q_values)))
        if terminated or truncated:
            break
    return env


def to_meters(lons, lats):
    meters_per_deg_lat = 111_320
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.radians(np.mean(lats)))
    return np.column_stack([np.asarray(lons) * meters_per_deg_lon, np.asarray(lats) * meters_per_deg_lat])


def relabel(labels):
    """Mark groups under MIN_ZONE_STALLS as -1 and number zones largest-first."""
    labels = np.asarray(labels)
    zone_ids = np.full(len(labels), -1)
    groups = [g for g in np.unique(labels) if g != -1 and (labels == g).sum() >= MIN_ZONE_STALLS]
    for zone_id, g in enumerate(sorted(groups, key=lambda g: -(labels == g).sum())):
        zone_ids[labels == g] = zone_id
    return zone_ids


def run(cluster_fn, algo_name, out_dir):
    wards = gpd.read_file("docs/mumbai_wards.geojson")
    boundary = unary_union(wards[wards["name"].isin(WARD_NAMES)].geometry)
    buildings = ox.features_from_polygon(boundary, tags={"building": True})
    roads_all = ox.features_from_polygon(boundary, tags={"highway": True})
    road_lines = roads_all[roads_all.geometry.type.isin(["LineString", "MultiLineString"])]
    minx, miny, maxx, maxy = boundary.bounds

    env = place_stalls()
    cell_size_deg = (maxx - minx) / env.footfall.shape[1]
    lons = np.array([minx + (j + 0.5) * cell_size_deg for i, j in env.placed_cells])
    lats = np.array([miny + (i + 0.5) * cell_size_deg for i, j in env.placed_cells])

    zone_ids = relabel(cluster_fn(to_meters(lons, lats)))
    n_zones = int(zone_ids.max()) + 1
    n_isolated = int((zone_ids == -1).sum())
    print(f"[{algo_name}] {n_zones} zones, {len(lons) - n_isolated} stalls in zones, {n_isolated} isolated")

    def draw_base(ax):
        buildings.plot(ax=ax, color="#c9c9c9", edgecolor="none", zorder=1)
        road_lines.plot(ax=ax, color="#555555", linewidth=0.6, zorder=2)
        gpd.GeoSeries([boundary]).boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=3)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_xticks([])
        ax.set_yticks([])

    def draw_zones(ax):
        draw_base(ax)
        cmap = plt.get_cmap("tab20", max(n_zones, 1))
        for z in range(n_zones):
            in_zone = zone_ids == z
            color = cmap(z)
            hull = MultiPoint(list(zip(lons[in_zone], lats[in_zone]))).convex_hull.buffer(cell_size_deg)
            gpd.GeoSeries([hull]).plot(ax=ax, facecolor=color, edgecolor=color, alpha=0.3, linewidth=1.5, zorder=4)
            ax.scatter(lons[in_zone], lats[in_zone], color=color, s=14, marker="s", zorder=5)
            ax.annotate(f"Z{z + 1} ({int(in_zone.sum())})", (hull.centroid.x, hull.centroid.y),
                        fontsize=8, fontweight="bold", ha="center", zorder=6)
        isolated = zone_ids == -1
        if isolated.any():
            ax.scatter(lons[isolated], lats[isolated], c="black", s=14, marker="x",
                       label=f"Isolated stall ({n_isolated})", zorder=5)
            ax.legend()
        ax.set_title(f"{algo_name} | {n_zones} zones from {len(lons)} stalls")

    def draw_legal(ax):
        draw_base(ax)
        ax.imshow(np.ma.masked_where(env.legal == 0, env.legal), origin="lower",
                  extent=(minx, maxx, miny, maxy), cmap="Greens", vmin=0, vmax=1,
                  interpolation="none", alpha=0.45, zorder=2.5)
        ax.set_title(f"Legal Mask Over Map | {int(env.legal.sum())} cells")

    fig, ax = plt.subplots(figsize=(10, 10))
    draw_zones(ax)
    fig.tight_layout()
    path = os.path.join(out_dir, "result_cluster.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")

    eval_fig, eval_axes = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)
    draw_zones(eval_axes[0])
    draw_legal(eval_axes[1])
    eval_fig.suptitle(f"{' + '.join(WARD_NAMES)} | {algo_name} | stalls = {len(lons)} | cell size = {CELL_SIZE_DEG} degrees")
    path = os.path.join(out_dir, "result_cluster_eval.png")
    eval_fig.savefig(path, dpi=150)
    print(f"Saved: {path}")

    plt.show()
