"""Vijayawada settings for running the Mumbai sliding-window model (inference only).

The scripts in this folder reuse sliding_window/src; this file is the only
Vijayawada-specific part.
"""

import os

import geopandas as gpd
import osmnx as ox
from shapely.ops import unary_union

VJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(VJ_ROOT)
BOUNDARY_GEOJSON = os.path.join(VJ_ROOT, "data", "vijayawada_boundary.geojson")
GEOCODE_QUERY = "Vijayawada, Andhra Pradesh, India"

CITY_TAG = "vijayawada"
# These must match the Mumbai model: the same cell size and window shape give
# the network the same input and action shape, and a cell stays ~28m in both cities.
CELL_SIZE_DEG = 0.00025
WINDOW_CELLS = 400
STRIDE_CELLS = 400
SMOOTHING_RADIUS_METERS = 75
N_STALLS = 100  # default stalls placed per window
MIN_LEGAL_CELLS = N_STALLS * 2
TRAIN_WINDOWS = None  # inference only: use every built window

RUN_TAG = f"{CITY_TAG}_{format(CELL_SIZE_DEG, 'g')}_w{WINDOW_CELLS}"
GRID_DIR = os.path.join(VJ_ROOT, "footfall_grid_data", RUN_TAG)
MANIFEST_PATH = os.path.join(GRID_DIR, "windows.json")
# The model trained on Mumbai, used as-is.
MODEL_PATH = os.path.join(PROJECT_ROOT, "sliding_window", "models",
                          f"hawker_dqn_mumbai_{format(CELL_SIZE_DEG, 'g')}_w{WINDOW_CELLS}_stalls100.zip")
VIS_DIR = os.path.join(VJ_ROOT, "visualizations", RUN_TAG)


def load_city():
    """Return (boundary GeoDataFrame, boundary geometry).

    The municipal boundary comes from OpenStreetMap, fetched once and saved,
    so later runs don't depend on the geocoding service.
    """
    if not os.path.exists(BOUNDARY_GEOJSON):
        os.makedirs(os.path.dirname(BOUNDARY_GEOJSON), exist_ok=True)
        city = ox.geocode_to_gdf(GEOCODE_QUERY)[["geometry"]].assign(name="Vijayawada")
        city.to_file(BOUNDARY_GEOJSON, driver="GeoJSON")
    wards = gpd.read_file(BOUNDARY_GEOJSON)
    return wards, unary_union(wards.geometry)


def window_path(window_id):
    return os.path.join(GRID_DIR, f"window_{window_id}.npz")
