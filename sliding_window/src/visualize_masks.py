"""Stitch every built window's masks into one city-wide view.

Reads only the saved window .npz files, so it's fast and needs no OSM
download. Dashed boxes show the windows.

    python sliding_window/src/visualize_masks.py
    python sliding_window/src/visualize_masks.py --window r00_c00
"""

import argparse
import json
import os

import geopandas as gpd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import config

LAYERS = [
    ("Buildings", "building", "Reds"),
    ("Road carriageways", "road", "Oranges"),
    ("Footways / pedestrian paths", "footway", "Blues"),
    ("Water / wetlands", "water", "PuBu"),
    ("Railways", "railway", "Purples"),
    ("Final legal mask", "legal", "Greens"),
    ("Footfall (smoothed, per-window normalised)", "footfall", "YlOrRd"),
]


def stitch(manifest, key):
    """Place each window's array onto one city-wide canvas."""
    w, s = manifest["window_cells"], manifest["stride_cells"]
    wins = manifest["windows"]
    n_rows = max(x["row"] for x in wins) * s + w
    n_cols = max(x["col"] for x in wins) * s + w
    canvas = np.zeros((n_rows, n_cols), dtype=np.float32)
    for win in wins:
        if win["built"]:
            r0, c0 = win["row"] * s, win["col"] * s
            canvas[r0:r0 + w, c0:c0 + w] = np.load(config.window_path(win["id"]))[key]
    return canvas


def draw_windows(ax, manifest):
    w, cs = manifest["window_cells"], manifest["cell_size_deg"]
    for win in manifest["windows"]:
        color = "black" if win["trainable"] else "#999999"
        ax.add_patch(patches.Rectangle((win["origin_lon"], win["origin_lat"]), w * cs, w * cs,
                                       fill=False, edgecolor=color, linestyle="--", linewidth=0.8))
        ax.text(win["origin_lon"] + w * cs / 2, win["origin_lat"] + w * cs / 2, win["id"],
                fontsize=6, color=color, ha="center", va="center", clip_on=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", nargs="*", help="Only show these window ids, e.g. r00_c00 (default: all built)")
    args = parser.parse_args()

    with open(config.MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    if args.window:
        for win in manifest["windows"]:
            win["built"] = win["built"] and win["id"] in args.window
    wards, _ = config.load_city()
    cs = manifest["cell_size_deg"]
    origin_lon = min(w["origin_lon"] for w in manifest["windows"])
    origin_lat = min(w["origin_lat"] for w in manifest["windows"])

    os.makedirs(config.VIS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(24, 22), constrained_layout=True)
    stats = {}
    for ax, (title, key, cmap) in zip(axes.flat, LAYERS):
        canvas = stitch(manifest, key)
        extent = (origin_lon, origin_lon + canvas.shape[1] * cs, origin_lat, origin_lat + canvas.shape[0] * cs)
        shown = canvas if key == "footfall" else np.ma.masked_where(canvas == 0, canvas)
        ax.imshow(shown, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=1, interpolation="none")
        wards.boundary.plot(ax=ax, color="black", linewidth=0.4)
        draw_windows(ax, manifest)
        if key != "footfall":
            stats[f"{key}_cells"] = int(canvas.sum())
            title = f"{title} | {int(canvas.sum())} cells"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    axes.flat[-1].axis("off")

    built = sum(w["built"] for w in manifest["windows"])
    trainable = sum(w["trainable"] for w in manifest["windows"])
    fig.suptitle(f"{config.CITY_TAG} | {built} windows built, {trainable} trainable | "
                 f"{manifest['window_cells']}x{manifest['window_cells']} cells @ {cs} deg", fontsize=16)
    path = os.path.join(config.VIS_DIR, "masks_visualization.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")

    stats.update({"windows_built": built, "windows_trainable": trainable,
                  "window_cells": manifest["window_cells"], "cell_size_degrees": cs})
    stats_path = os.path.join(config.VIS_DIR, "mask_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_path}")
    plt.show()


if __name__ == "__main__":
    main()
