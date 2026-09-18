"""Visualize all Mumbai wards together, each outlined by its bounding box and labeled.

Example:
    python src/visualize_all_wards.py
"""

import os

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = os.path.join("visualizations")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "all_wards_map.png")


def plot_all_wards():
    wards = gpd.read_file("docs/mumbai_wards.geojson")

    fig, ax = plt.subplots(figsize=(14, 16))
    cmap = plt.get_cmap("tab20", len(wards))

    for idx, row in wards.reset_index(drop=True).iterrows():
        geom = row.geometry
        name = row["name"]
        color = cmap(idx)

        gpd.GeoSeries([geom]).plot(ax=ax, facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.6)

        minx, miny, maxx, maxy = geom.bounds
        ax.add_patch(
            patches.Rectangle(
                (minx, miny),
                maxx - minx,
                maxy - miny,
                fill=False,
                edgecolor="black",
                linewidth=0.8,
                linestyle="--",
            )
        )

        cx, cy = geom.centroid.x, geom.centroid.y
        ax.text(cx, cy, name, fontsize=9, fontweight="bold", ha="center", va="center")

    all_minx, all_miny, all_maxx, all_maxy = wards.total_bounds
    ax.set_xlim(all_minx, all_maxx)
    ax.set_ylim(all_miny, all_maxy)
    ax.set_title(f"Greater Mumbai — {len(wards)} wards, each boxed and labeled")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    plot_all_wards()
