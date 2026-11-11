"""Run the trained model on every window and plot all stalls on one city map.

Each window gets its own PLACE_STALLS placements (one zone recommendation per
window), so every part of the city gets nearby stalls.

    python sliding_window/src/visualize_real.py
    python sliding_window/src/visualize_real.py --cluster
    python sliding_window/src/visualize_real.py --window r00_c00
"""

import argparse
import json
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from shapely.geometry import MultiPoint, box
from shapely.ops import unary_union
from stable_baselines3 import DQN

import config
from build_footfall_grid import safe_features
from hawker_env import HawkerZoneEnv
from visualize_masks import draw_windows, stitch

parser = argparse.ArgumentParser()
parser.add_argument("--stalls", type=int, default=config.N_STALLS, help="Stalls to place per window")
parser.add_argument("--cluster", action="store_true", help="Group placed stalls into hawker zones")
parser.add_argument("--cluster-eps-meters", type=float, default=400,
                    help="DBSCAN neighbourhood radius in metres (default: 400, ~5 min walk)")
parser.add_argument("--cluster-min-samples", type=int, default=3,
                    help="DBSCAN: stalls needed within the radius to start a zone (default: 3)")
parser.add_argument("--window", nargs="*", help="Only place stalls in these window ids, e.g. r00_c00 (default: all)")
args = parser.parse_args()
MIN_ZONE_STALLS = 3

with open(config.MANIFEST_PATH, encoding="utf-8") as f:
    manifest = json.load(f)
if not os.path.exists(config.MODEL_PATH):
    raise SystemExit(f"No model at {config.MODEL_PATH}: run train_cnn.py first.")
model = DQN.load(config.MODEL_PATH)
wards, city = config.load_city()
cs = manifest["cell_size_deg"]

stall_lons, stall_lats = [], []
# TRAIN_WINDOWS is None for inference-only cities: use every built window.
selected = args.window or config.TRAIN_WINDOWS or [w["id"] for w in manifest["windows"]]
for win in [w for w in manifest["windows"] if w["trainable"] and w["id"] in selected]:
    env = HawkerZoneEnv(n_stalls=args.stalls, max_steps=args.stalls)
    env.load_real_footfall(config.window_path(win["id"]))
    obs, _ = env.reset()
    for _ in range(env.n_stalls):
        q_values = model.q_net(model.policy.obs_to_tensor(obs)[0]).detach().cpu().numpy()[0]
        q_values[(env.legal.flatten() == 0) | (env.occupied.flatten() == 1)] = -np.inf
        obs, _, terminated, truncated, _ = env.step(int(np.argmax(q_values)))
        if terminated or truncated:
            break
    # Exact position from the window's stored origin, not a re-derived cell size.
    stall_lons += [win["origin_lon"] + (j + 0.5) * cs for i, j in env.placed_cells]
    stall_lats += [win["origin_lat"] + (i + 0.5) * cs for i, j in env.placed_cells]
    print(f"  {win['id']}: placed {len(env.placed_cells)} stalls")
stall_lons, stall_lats = np.array(stall_lons), np.array(stall_lats)
print(f"Placed {len(stall_lons)} stalls across the city")


def cluster_stalls():
    meters_per_deg_lat = 111_320
    xy = np.column_stack([stall_lons * meters_per_deg_lat * np.cos(np.radians(stall_lats.mean())),
                          stall_lats * meters_per_deg_lat])
    # DBSCAN: a zone grows from stalls with enough neighbours within eps;
    # stalls in sparse areas come back as -1 (noise) and are shown as isolated.
    raw = DBSCAN(eps=args.cluster_eps_meters, min_samples=args.cluster_min_samples).fit_predict(xy)
    zone_ids = np.full(len(raw), -1)
    sizes = {c: int((raw == c).sum()) for c in np.unique(raw) if c != -1}
    for zone_id, c in enumerate(sorted((c for c in sizes if sizes[c] >= MIN_ZONE_STALLS), key=lambda c: -sizes[c])):
        zone_ids[raw == c] = zone_id
    return zone_ids


zone_ids = cluster_stalls() if args.cluster and len(stall_lons) > 1 else None

# Street map underneath: each window's buildings and roads, queried with the
# exact area the build step used so they come from the osmnx cache instead of
# a fresh download. --window zooms to just those windows.
shown_windows = [w for w in manifest["windows"] if w["trainable"] and w["id"] in selected]
side = config.WINDOW_CELLS * cs
areas = [box(w["origin_lon"], w["origin_lat"], w["origin_lon"] + side, w["origin_lat"] + side).intersection(city)
         for w in shown_windows]
print("Loading street map for", len(areas), "windows...", flush=True)
buildings = pd.concat([safe_features(a, {"building": True}) for a in areas])
roads = pd.concat([safe_features(a, {"highway": True}) for a in areas])
road_lines = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])] if not roads.empty else roads

minx, miny, maxx, maxy = unary_union(areas).bounds if args.window else city.bounds
# Markers keep the same size relative to the streets: 12 when the plot spans
# one window, shrinking with the square of how much wider the view is.
marker_size = 12 * min(1.0, side / (maxx - minx)) ** 2
# Thinner lines city-wide, otherwise dense neighbourhoods turn solid grey.
road_width = 0.6 if args.window else 0.15


def draw_base(ax):
    if not buildings.empty:
        buildings.plot(ax=ax, color="#c9c9c9", edgecolor="none", zorder=1)
    if not road_lines.empty:
        road_lines.plot(ax=ax, color="#555555", linewidth=road_width, zorder=2)
    wards.boundary.plot(ax=ax, color="#555555", linewidth=0.6, zorder=2)
    gpd.GeoSeries([city]).boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=3)
    if not args.window:
        draw_windows(ax, manifest)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_placement_panel(ax):
    draw_base(ax)
    if zone_ids is None:
        ax.scatter(stall_lons, stall_lats, c="green", s=marker_size, marker="s", label="Vendor stall", zorder=4)
        ax.set_title(f"Vendor Placement | {len(stall_lons)} stalls")
    else:
        n_zones = int(zone_ids.max()) + 1
        cmap = plt.get_cmap("tab20", max(n_zones, 1))
        for z in range(n_zones):
            in_zone = zone_ids == z
            hull = MultiPoint(list(zip(stall_lons[in_zone], stall_lats[in_zone]))).convex_hull.buffer(cs)
            gpd.GeoSeries([hull]).plot(ax=ax, facecolor=cmap(z), edgecolor=cmap(z), alpha=0.35, zorder=4)
            if args.window:  # zone labels are readable only when zoomed in
                ax.annotate(f"Z{z + 1} ({int(in_zone.sum())})", (hull.centroid.x, hull.centroid.y),
                            fontsize=8, fontweight="bold", ha="center", zorder=6)
            ax.scatter(stall_lons[in_zone], stall_lats[in_zone], color=cmap(z), s=marker_size, marker="s", zorder=5)
        isolated = zone_ids == -1
        if isolated.any():
            ax.scatter(stall_lons[isolated], stall_lats[isolated], c="black", s=marker_size, marker="x",
                       label=f"Isolated stall ({int(isolated.sum())})", zorder=5)
        ax.set_title(f"Hawker Zones (DBSCAN) | {n_zones} zones from {len(stall_lons)} stalls")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left")


def draw_legal_panel(ax):
    draw_base(ax)
    legal = stitch(manifest, "legal")
    origin_lon = min(w["origin_lon"] for w in manifest["windows"])
    origin_lat = min(w["origin_lat"] for w in manifest["windows"])
    ax.imshow(np.ma.masked_where(legal == 0, legal), origin="lower", cmap="Greens", vmin=0, vmax=1,
              extent=(origin_lon, origin_lon + legal.shape[1] * cs, origin_lat, origin_lat + legal.shape[0] * cs),
              interpolation="none", alpha=0.6, zorder=2.5)
    ax.set_title(f"Legal Mask | {sum(w['legal_cells'] for w in shown_windows):,} cells")


results_dir = os.path.join(config.VIS_DIR, "results")
os.makedirs(results_dir, exist_ok=True)
result_name = "result_cluster" if zone_ids is not None else "result"
if args.window:
    result_name += "_" + "_".join(args.window)

# Size figures to the area's shape: square-ish for one window, tall for the city.
aspect = (maxy - miny) / (maxx - minx)
fig, ax = plt.subplots(figsize=(10, 10 * aspect))
draw_placement_panel(ax)
fig.tight_layout()
path = os.path.join(results_dir, f"{result_name}.png")
fig.savefig(path, dpi=150)
print(f"Saved: {path}")

eval_fig, eval_axes = plt.subplots(1, 2, figsize=(20, 10 * aspect), constrained_layout=True)
draw_placement_panel(eval_axes[0])
draw_legal_panel(eval_axes[1])
eval_fig.suptitle(f"{config.CITY_TAG} | {args.stalls} stalls per window | "
                  f"{config.WINDOW_CELLS}x{config.WINDOW_CELLS} windows @ {cs} deg")
path = os.path.join(results_dir, f"{result_name}_eval.png")
eval_fig.savefig(path, dpi=150)
print(f"Saved: {path}")

plt.show()
