"""Build one fixed-size grid per window, sliding across the whole city.

Each window is WINDOW_CELLS x WINDOW_CELLS cells on a single city-wide grid.
Cells outside the city boundary are marked illegal, so every window has the
same shape whether it is fully inland or mostly sea.

Resumable: windows whose .npz already exists are skipped unless --force.

    python sliding_window/src/build_footfall_grid.py               # all windows
    python sliding_window/src/build_footfall_grid.py --only r01_c01 # just one
"""

import argparse
import json
import math
import os
import time

import geopandas as gpd
import numpy as np
import osmnx as ox
import requests
import shapely
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError
from scipy.ndimage import gaussian_filter
from shapely.geometry import box

import config

BLOCKED_ROAD_TYPES = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "unclassified", "residential", "service",
    "living_street",
}
FOOTWAY_TYPES = {"footway", "path", "pedestrian", "cycleway", "steps", "corridor"}


OVERPASS_MIRRORS = [
    "https://overpass-api.de/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
]
# osmnx reads the server's /status page before every query and sleeps 60s
# whenever it can't, which both an overloaded main server and the mirrors
# cause. Busy servers are handled by switching mirrors in safe_features instead.
ox.settings.overpass_rate_limit = False

# osmnx keys its cache on server URL + query, so data fetched through one
# mirror would be re-downloaded when another mirror is active. Also look the
# query up under every other mirror's URL before going to the network.
_check_cache_one_server = ox._http._check_cache


def _check_cache_any_mirror(key):
    hit = _check_cache_one_server(key)
    current = next((m for m in OVERPASS_MIRRORS if key.startswith(m + "/")), None)
    for other in OVERPASS_MIRRORS:
        if hit is not None or current is None:
            break
        if other != current:
            hit = _check_cache_one_server(other + key[len(current):])
    return hit


ox._http._check_cache = _check_cache_any_mirror


def _responds(base_url):
    """Quick check: does this Overpass server answer a tiny query? 10s to connect, 30s to reply."""
    try:
        r = requests.post(base_url + "/interpreter", data={"data": "[out:json];node(1);out;"}, timeout=(10, 30))
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _switch_to_responding_mirror():
    """Cycle through the mirrors after the current one until one answers."""
    start = OVERPASS_MIRRORS.index(ox.settings.overpass_url) + 1 if ox.settings.overpass_url in OVERPASS_MIRRORS else 0
    while True:
        for k in range(len(OVERPASS_MIRRORS)):
            url = OVERPASS_MIRRORS[(start + k) % len(OVERPASS_MIRRORS)]
            if _responds(url):
                print(f"    switched to {url}", flush=True)
                ox.settings.overpass_url = url
                return
        print("    no OSM server answering, checking again in 3s", flush=True)
        time.sleep(3)


def safe_features(polygon, tags):
    """OSM features in polygon (empty if none). Keeps retrying until a server answers."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return ox.features_from_polygon(polygon, tags=tags)
        except InsufficientResponseError:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        except (ResponseStatusCodeError, requests.exceptions.RequestException) as e:
            # Completed sub-queries are cached, so a retry only re-downloads the part that failed.
            print(f"    {ox.settings.overpass_url} failed ({type(e).__name__}, attempt {attempt}), finding a server...", flush=True)
            _switch_to_responding_mirror()


def has_type(value, accepted):
    values = value if isinstance(value, list) else [value]
    return any(v in accepted for v in values)


def plan_windows(city):
    """Every window position covering the city's bounding box, row-major from the south-west."""
    minx, miny, maxx, maxy = city.bounds
    cs, w, s = config.CELL_SIZE_DEG, config.WINDOW_CELLS, config.STRIDE_CELLS
    total_rows = math.ceil((maxy - miny) / cs)
    total_cols = math.ceil((maxx - minx) / cs)
    n_rows = 1 + math.ceil(max(0, total_rows - w) / s)
    n_cols = 1 + math.ceil(max(0, total_cols - w) / s)
    windows = []
    for r in range(n_rows):
        for c in range(n_cols):
            windows.append({
                "id": f"r{r:02d}_c{c:02d}",
                "row": r,
                "col": c,
                "origin_lon": minx + c * s * cs,
                "origin_lat": miny + r * s * cs,
            })
    return windows


def build_window(win, city):
    cs, w = config.CELL_SIZE_DEG, config.WINDOW_CELLS
    x0, y0 = win["origin_lon"], win["origin_lat"]
    window_box = box(x0, y0, x0 + w * cs, y0 + w * cs)
    area = window_box.intersection(city)
    if area.is_empty:
        return False

    # Cell classification (same rules as the main pipeline). Anything outside
    # the city boundary stays 0 in every mask, i.e. illegal.
    rr, cc = np.meshgrid(np.arange(w), np.arange(w), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    centers = shapely.points(x0 + (cc + 0.5) * cs, y0 + (rr + 0.5) * cs)
    inside = shapely.contains(city, centers)

    inside_mask = np.zeros((w, w), dtype=np.float32)
    inside_mask[rr[inside], cc[inside]] = 1
    masks = {k: np.zeros((w, w), dtype=np.float32) for k in ("building", "road", "footway", "water", "railway")}

    if inside.any():
        in_r, in_c = rr[inside], cc[inside]
        cells = gpd.GeoDataFrame(
            {"row": in_r, "col": in_c},
            geometry=shapely.box(x0 + in_c * cs, y0 + in_r * cs, x0 + (in_c + 1) * cs, y0 + (in_r + 1) * cs),
            crs="EPSG:4326",
        )

        buildings = safe_features(area, {"building": True})
        roads = safe_features(area, {"highway": True})
        if not roads.empty:
            roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]
        water = safe_features(area, {"natural": ["water", "wetland"], "water": True})
        railways = safe_features(area, {"railway": True})

        layers = {
            "building": buildings,
            "road": roads[roads["highway"].apply(lambda v: has_type(v, BLOCKED_ROAD_TYPES))] if not roads.empty else roads,
            "footway": roads[roads["highway"].apply(lambda v: has_type(v, FOOTWAY_TYPES))] if not roads.empty else roads,
            "water": water,
            "railway": railways,
        }
        for name, layer in layers.items():
            if layer.empty:
                continue
            hits = gpd.sjoin(cells, layer[["geometry"]], predicate="intersects", how="inner")
            masks[name][hits["row"].to_numpy(), hits["col"].to_numpy()] = 1

    legal = inside_mask.copy()
    legal[(masks["building"] == 1) | (masks["road"] == 1) | (masks["water"] == 1) | (masks["railway"] == 1)] = 0
    legal[(masks["footway"] == 1) & (masks["road"] == 0) & (masks["building"] == 0)
          & (masks["water"] == 0) & (masks["railway"] == 0)] = 1

    # Footfall is counted on a padded grid so shops just outside this window
    # still spread into its edge cells, then cropped back to the window.
    cell_size_meters = cs * 111_320
    sigma = config.SMOOTHING_RADIUS_METERS / cell_size_meters
    pad = math.ceil(3 * sigma)
    pois = safe_features(window_box.buffer(pad * cs).intersection(city), {"shop": True, "amenity": True})
    padded = np.zeros((w + 2 * pad, w + 2 * pad))
    if not pois.empty:
        pois = pois[pois.geometry.type == "Point"]
        p_col = np.floor((pois.geometry.x.to_numpy() - (x0 - pad * cs)) / cs).astype(int)
        p_row = np.floor((pois.geometry.y.to_numpy() - (y0 - pad * cs)) / cs).astype(int)
        valid = (p_row >= 0) & (p_row < w + 2 * pad) & (p_col >= 0) & (p_col < w + 2 * pad)
        np.add.at(padded, (p_row[valid], p_col[valid]), 1)
    footfall = gaussian_filter(padded, sigma=sigma)[pad:pad + w, pad:pad + w]
    if footfall.max() > 0:
        footfall = footfall / footfall.max()

    np.savez(
        config.window_path(win["id"]),
        footfall=footfall.astype(np.float32),
        legal=legal,
        inside=inside_mask,
        **masks,
        origin_lon=x0,
        origin_lat=y0,
        cell_size_deg=cs,
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="Window ids to build, e.g. r01_c01 (default: all)")
    parser.add_argument("--force", action="store_true", help="Rebuild windows that already exist")
    args = parser.parse_args()

    os.makedirs(config.GRID_DIR, exist_ok=True)
    _, city = config.load_city()
    shapely.prepare(city)
    windows = plan_windows(city)
    print(f"{len(windows)} windows of {config.WINDOW_CELLS}x{config.WINDOW_CELLS} cells planned")

    for win in windows:
        path = config.window_path(win["id"])
        if args.only and win["id"] not in args.only:
            pass
        elif os.path.exists(path) and not args.force:
            print(f"  {win['id']}: already built")
        else:
            print(f"  {win['id']}: building...", flush=True)
            if not build_window(win, city):
                print(f"  {win['id']}: outside the city, skipped")

        if os.path.exists(path):
            data = np.load(path)
            win["built"] = True
            win["inside_cells"] = int(data["inside"].sum())
            win["legal_cells"] = int(data["legal"].sum())
            win["trainable"] = win["legal_cells"] >= config.MIN_LEGAL_CELLS
        else:
            win["built"] = False
            win["trainable"] = False

    manifest = {
        "city": config.CITY_TAG,
        "cell_size_deg": config.CELL_SIZE_DEG,
        "window_cells": config.WINDOW_CELLS,
        "stride_cells": config.STRIDE_CELLS,
        "windows": windows,
    }
    with open(config.MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    built = [w for w in windows if w["built"]]
    trainable = [w for w in windows if w["trainable"]]
    print(f"Saved: {config.MANIFEST_PATH}")
    print(f"{len(built)} built, {len(trainable)} trainable (>= {config.MIN_LEGAL_CELLS} legal cells)")
    for w in built:
        print(f"  {w['id']}: {w['legal_cells']} legal / {w['inside_cells']} inside city"
              + ("" if w["trainable"] else "  [too sparse, not trained]"))


if __name__ == "__main__":
    main()
