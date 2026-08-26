"""Visualize physical placement masks at a chosen grid resolution.

Examples:
    python src/visualize_masks.py
    python src/visualize_masks.py --cell-size 0.0002 --output docs/masks_20m.png
"""

import argparse
import json
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from shapely.geometry import Point, box


FOOTWAY_TYPES = {"footway", "path", "pedestrian", "cycleway", "steps", "corridor"}
BLOCKED_ROAD_TYPES = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "unclassified", "residential", "service", "living_street",
}


def has_type(value, accepted):
    values = value if isinstance(value, list) else [value]
    return any(item in accepted for item in values)


def build_masks(cell_size):
    wards = gpd.read_file("docs/mumbai_wards.geojson")
    ward = wards[wards["name"] == "G/N"]
    if ward.empty:
        raise ValueError("Could not find ward named G/N")
    boundary = ward.geometry.iloc[0]

    buildings = ox.features_from_polygon(boundary, tags={"building": True})
    roads = ox.features_from_polygon(boundary, tags={"highway": True})
    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]
    water = ox.features_from_polygon(
        boundary, tags={"natural": ["water", "wetland"], "water": True}
    )
    railways = ox.features_from_polygon(boundary, tags={"railway": True})

    minx, miny, maxx, maxy = boundary.bounds
    cols = int((maxx - minx) / cell_size) + 1
    rows = int((maxy - miny) / cell_size) + 1

    building_mask = np.zeros((rows, cols), dtype=np.float32)
    road_mask = np.zeros((rows, cols), dtype=np.float32)
    water_mask = np.zeros((rows, cols), dtype=np.float32)
    railway_mask = np.zeros((rows, cols), dtype=np.float32)
    footway_mask = np.zeros((rows, cols), dtype=np.float32)
    legal_mask = np.zeros((rows, cols), dtype=np.float32)

    blocked_roads = roads[roads["highway"].apply(lambda x: has_type(x, BLOCKED_ROAD_TYPES))]
    footways = roads[roads["highway"].apply(lambda x: has_type(x, FOOTWAY_TYPES))]

    for i in range(rows):
        for j in range(cols):
            point = Point(
                minx + (j + 0.5) * cell_size,
                miny + (i + 0.5) * cell_size,
            )
            cell_polygon = box(
                minx + j * cell_size,
                miny + i * cell_size,
                minx + (j + 1) * cell_size,
                miny + (i + 1) * cell_size,
            )
            if not boundary.contains(point):
                continue

            legal_mask[i, j] = 1
            building_mask[i, j] = buildings.geometry.intersects(cell_polygon).any()
            road_mask[i, j] = blocked_roads.geometry.intersects(cell_polygon).any()
            water_mask[i, j] = water.geometry.intersects(cell_polygon).any()
            railway_mask[i, j] = railways.geometry.intersects(cell_polygon).any()
            footway_mask[i, j] = footways.geometry.intersects(cell_polygon).any()

    legal_mask[
        (building_mask == 1)
        | (road_mask == 1)
        | (water_mask == 1)
        | (railway_mask == 1)
    ] = 0

    return boundary, buildings, blocked_roads, footways, water, railways, {
        "building": building_mask,
        "road": road_mask,
        "water": water_mask,
        "railway": railway_mask,
        "footway": footway_mask,
        "legal": legal_mask,
    }, (minx, miny, maxx, maxy)


def plot_masks(cell_size, output, overlay_output):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(overlay_output) or ".", exist_ok=True)
    boundary, buildings, roads, footways, water, railways, masks, bounds = build_masks(cell_size)
    minx, miny, maxx, maxy = bounds
    panels = [
        ("Buildings", masks["building"], "Reds"),
        ("Road carriageways", masks["road"], "Oranges"),
        ("Footways / pedestrian paths", masks["footway"], "Blues"),
        ("Water / wetlands", masks["water"], "PuBu"),
        ("Railways", masks["railway"], "Purples"),
        ("Final legal mask", masks["legal"], "Greens"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True)
    for ax, (title, mask, cmap) in zip(axes.flat, panels):
        ax.imshow(
            mask,
            origin="lower",
            extent=(minx, maxx, miny, maxy),
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="none",
            alpha=0.85,
        )
        gpd.GeoSeries([boundary.boundary]).plot(ax=ax, color="black", linewidth=1)
        ax.set_title(f"{title} | {int(mask.sum())} cells")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.suptitle(
        f"G/North physical masks | cell size = {cell_size} degrees | "
        f"grid = {masks['legal'].shape[0]} × {masks['legal'].shape[1]}"
    )
    fig.savefig(output, dpi=180)
    print(f"Saved: {output}")

    # Second view: four panels, each showing one mask over the original map.
    overlay_fig, overlay_axes = plt.subplots(2, 3, figsize=(19, 12), constrained_layout=True)
    overlay_panels = [
        ("Buildings mask", masks["building"], "Reds"),
        ("Road carriageway mask", masks["road"], "Oranges"),
        ("Footway mask", masks["footway"], "Blues"),
        ("Water / wetlands mask", masks["water"], "PuBu"),
        ("Railway mask", masks["railway"], "Purples"),
        ("Final legal mask", masks["legal"], "Greens"),
    ]
    for overlay_ax, (title, mask, cmap) in zip(overlay_axes.flat, overlay_panels):
        buildings.plot(ax=overlay_ax, facecolor="#bdbdbd", edgecolor="none", alpha=0.65)
        roads.plot(ax=overlay_ax, color="#333333", linewidth=0.55, alpha=0.65)
        if title == "Footway mask":
            footways.plot(ax=overlay_ax, color="#1976d2", linewidth=0.8, alpha=0.8)
        if title == "Water / wetlands mask":
            water.plot(ax=overlay_ax, facecolor="#80deea", edgecolor="none", alpha=0.65)
        if title == "Railway mask":
            railways.plot(ax=overlay_ax, color="#7b1fa2", linewidth=1.0, alpha=0.8)
        overlay_ax.imshow(
            np.ma.masked_where(mask == 0, mask),
            origin="lower",
            extent=(minx, maxx, miny, maxy),
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="none",
            alpha=0.42,
        )
        gpd.GeoSeries([boundary.boundary]).plot(ax=overlay_ax, color="black", linewidth=1)
        overlay_ax.set_xlim(minx, maxx)
        overlay_ax.set_ylim(miny, maxy)
        overlay_ax.set_title(f"{title} over original map | {int(mask.sum())} cells")
        overlay_ax.set_xlabel("Longitude")
        overlay_ax.set_ylabel("Latitude")

    overlay_fig.suptitle(
        f"G/North mask overlays | cell size = {cell_size} degrees",
        fontsize=14,
    )
    overlay_fig.savefig(overlay_output, dpi=180)
    print(f"Saved: {overlay_output}")

    print(f"Grid shape: {masks['legal'].shape}")
    stats = {
        "cell_size_degrees": cell_size,
        "grid_shape": list(masks["legal"].shape),
        "total_states": int(masks["legal"].shape[0] * masks["legal"].shape[1]),
        "building_cells": int(masks["building"].sum()),
        "road_cells": int(masks["road"].sum()),
        "water_cells": int(masks["water"].sum()),
        "railway_cells": int(masks["railway"].sum()),
        "footway_cells": int(masks["footway"].sum()),
        "legal_cells": int(masks["legal"].sum()),
    }
    stats_path = os.path.join(os.path.dirname(output) or ".", "mask_stats.json")
    with open(stats_path, "w", encoding="utf-8") as stats_file:
        json.dump(stats, stats_file, indent=2)
    print(f"Saved: {stats_path}")
    for name, mask in masks.items():
        print(f"{name} cells: {int(mask.sum())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-size",
        type=float,
        default=0.00045,
        help="Cell width/height in degrees (default: 0.00045)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path",
    )
    parser.add_argument(
        "--overlay-output",
        default=None,
        help="Output path for the mask overlay map",
    )
    args = parser.parse_args()
    if args.cell_size <= 0:
        parser.error("--cell-size must be greater than zero")
    cell_size_label = format(args.cell_size, "g")
    output_dir = os.path.join("visualizations", f"cell_size_{cell_size_label}")
    output = args.output or os.path.join(output_dir, "masks_visualization.png")
    overlay_output = args.overlay_output or os.path.join(output_dir, "masks_overlay_map.png")
    plot_masks(args.cell_size, output, overlay_output)
