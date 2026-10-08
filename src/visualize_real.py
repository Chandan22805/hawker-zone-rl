import argparse
import os
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from shapely.geometry import MultiPoint
from shapely.ops import unary_union
from stable_baselines3 import DQN
from hawker_env import HawkerZoneEnv

parser = argparse.ArgumentParser()
parser.add_argument("--cluster", action="store_true", help="Group placed stalls into hawker zones")
parser.add_argument("--cluster-link-meters", type=float, default=400,
                    help="Stalls within this distance of another stall join the same zone (default: 400, ~5 min walk)")
args = parser.parse_args()

CLUSTER_LINK_METERS = args.cluster_link_meters
# Groups smaller than this are shown as isolated stalls, not zones.
MIN_ZONE_STALLS = 3

WARD_NAMES = ["G/N", "F/N", "G/S", "F/S"]
CELL_SIZE_DEG = 0.00025
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

TRAINED_STALLS = 100  # stall count the model file was trained/saved with
PLACE_STALLS = 200  # stall count to actually place at inference; can differ from TRAINED_STALLS

env = HawkerZoneEnv(n_stalls=PLACE_STALLS, max_steps=PLACE_STALLS)  # inference always picks valid cells, so steps == stalls placed
env.load_real_footfall(f"footfall_grid_data/{ward_tag}_data_{cell_size_label}.npz")
model = DQN.load(os.path.join("models", f"hawker_dqn_{ward_tag}_{cell_size_label}_stalls{TRAINED_STALLS}"))
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

cell_size_deg = (maxx - minx) / env.footfall.shape[1]
stall_lons = [minx + (j + 0.5) * cell_size_deg for i, j in env.placed_cells]
stall_lats = [miny + (i + 0.5) * cell_size_deg for i, j in env.placed_cells]


def cluster_stalls(lons, lats):
    """Single-linkage clustering in metres; returns a zone id per stall (-1 = isolated)."""
    lons, lats = np.asarray(lons), np.asarray(lats)
    if len(lons) < 2:
        return np.full(len(lons), -1)
    meters_per_deg_lat = 111_320
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.radians(lats.mean()))
    xy = np.column_stack([lons * meters_per_deg_lon, lats * meters_per_deg_lat])
    raw = fcluster(linkage(xy, method="single"), t=CLUSTER_LINK_METERS, criterion="distance")

    zone_ids = np.full(len(raw), -1)
    sizes = {c: int((raw == c).sum()) for c in np.unique(raw)}
    big = sorted((c for c in sizes if sizes[c] >= MIN_ZONE_STALLS), key=lambda c: -sizes[c])
    for zone_id, c in enumerate(big):
        zone_ids[raw == c] = zone_id
    return zone_ids


zone_ids = cluster_stalls(stall_lons, stall_lats) if args.cluster else None


def draw_stalls(ax):
    if zone_ids is None:
        ax.scatter(stall_lons, stall_lats, c="green", s=12, marker="s", label="Vendor stall", zorder=4)
        return

    lons, lats = np.asarray(stall_lons), np.asarray(stall_lats)
    n_zones = int(zone_ids.max()) + 1
    cmap = plt.get_cmap("tab20", max(n_zones, 1))
    for z in range(n_zones):
        in_zone = zone_ids == z
        color = cmap(z)
        hull = MultiPoint(list(zip(lons[in_zone], lats[in_zone]))).convex_hull.buffer(cell_size_deg)
        gpd.GeoSeries([hull]).plot(ax=ax, facecolor=color, edgecolor=color, alpha=0.3, linewidth=1.5, zorder=4)
        ax.scatter(lons[in_zone], lats[in_zone], color=color, s=14, marker="s", zorder=5)
        ax.annotate(
            f"Z{z + 1} ({int(in_zone.sum())})", (hull.centroid.x, hull.centroid.y),
            fontsize=8, fontweight="bold", ha="center", zorder=6,
        )
    isolated = zone_ids == -1
    if isolated.any():
        ax.scatter(lons[isolated], lats[isolated], c="black", s=14, marker="x",
                   label=f"Isolated stall ({int(isolated.sum())})", zorder=5)


def draw_placement_panel(ax):
    buildings.plot(ax=ax, color="#c9c9c9", edgecolor="none", zorder=1)
    road_lines.plot(ax=ax, color="#555555", linewidth=0.6, zorder=2)
    road_points.plot(ax=ax, color="#888888", markersize=3, zorder=2)
    gpd.GeoSeries([boundary_polygon]).boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=3)
    draw_stalls(ax)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    if zone_ids is None:
        ax.set_title("Vendor Placement on Real Map")
    else:
        ax.set_title(f"Hawker Zones | {int(zone_ids.max()) + 1} zones from {len(stall_lons)} stalls")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])


def draw_legal_mask_panel(ax):
    buildings.plot(ax=ax, color="#c9c9c9", edgecolor="none", zorder=1)
    road_lines.plot(ax=ax, color="#555555", linewidth=0.6, zorder=2)
    road_points.plot(ax=ax, color="#888888", markersize=3, zorder=2)
    gpd.GeoSeries([boundary_polygon]).boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=3)
    legal_mask = env.legal
    ax.imshow(
        np.ma.masked_where(legal_mask == 0, legal_mask),
        origin="lower",
        extent=(minx, maxx, miny, maxy),
        cmap="Greens",
        vmin=0,
        vmax=1,
        interpolation="none",
        alpha=0.45,
        zorder=2.5,
    )
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(f"Legal Mask Over Map | {int(legal_mask.sum())} cells")
    ax.set_xticks([])
    ax.set_yticks([])


results_dir = os.path.join(
    "visualizations", ward_tag, f"cell_size_{cell_size_label}", "results"
)
os.makedirs(results_dir, exist_ok=True)
result_name = "result_cluster" if args.cluster else "result"

fig, ax = plt.subplots(figsize=(10, 10))
draw_placement_panel(ax)
plt.tight_layout()
output_path = os.path.join(results_dir, f"{result_name}.png")
plt.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")

eval_fig, eval_axes = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)
draw_placement_panel(eval_axes[0])
draw_legal_mask_panel(eval_axes[1])
eval_fig.suptitle(f"{' + '.join(WARD_NAMES)} | stalls = {env.n_stalls} | cell size = {CELL_SIZE_DEG} degrees")
eval_output_path = os.path.join(results_dir, f"{result_name}_eval.png")
eval_fig.savefig(eval_output_path, dpi=150)
print(f"Saved: {eval_output_path}")

plt.show()