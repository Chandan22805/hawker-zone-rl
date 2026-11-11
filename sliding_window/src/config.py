"""Settings shared by every sliding-window script."""

import os

import geopandas as gpd
from shapely.ops import unary_union

SW_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SW_ROOT)
WARDS_GEOJSON = os.path.join(PROJECT_ROOT, "docs", "mumbai_wards.geojson")

CITY_TAG = "mumbai"
WARD_NAMES = None  # None = every ward in the geojson, i.e. the whole city
CELL_SIZE_DEG = 0.00025
WINDOW_CELLS = 400
STRIDE_CELLS = 400  # == WINDOW_CELLS means windows tile the city without overlap
N_STALLS = 100
SMOOTHING_RADIUS_METERS = 75
# A window needs room to place its stalls; sparser windows are skipped for training.
MIN_LEGAL_CELLS = N_STALLS * 2
# Windows to train on, in order: row by row from the south, west then east.
# Leaves out the near-empty sea windows.
TRAIN_WINDOWS = ["r00_c00", "r01_c00", "r01_c01", "r02_c00", "r02_c01", "r03_c00", "r03_c01"]

RUN_TAG = f"{CITY_TAG}_{format(CELL_SIZE_DEG, 'g')}_w{WINDOW_CELLS}"
GRID_DIR = os.path.join(SW_ROOT, "footfall_grid_data", RUN_TAG)
MANIFEST_PATH = os.path.join(GRID_DIR, "windows.json")
MODELS_DIR = os.path.join(SW_ROOT, "models")
# Explicit .zip: the "0.00025" in the name would otherwise make SB3 treat
# ".00025_w400..." as the extension and save without .zip.
MODEL_PATH = os.path.join(MODELS_DIR, f"hawker_dqn_{RUN_TAG}_stalls{N_STALLS}.zip")
PROGRESS_PATH = MODEL_PATH.replace(".zip", "_progress.json")
VIS_DIR = os.path.join(SW_ROOT, "visualizations", RUN_TAG)


def load_city():
    """Return (ward GeoDataFrame, unioned city boundary)."""
    wards = gpd.read_file(WARDS_GEOJSON)
    if WARD_NAMES is not None:
        wards = wards[wards["name"].isin(WARD_NAMES)]
    return wards, unary_union(wards.geometry)


def window_path(window_id):
    return os.path.join(GRID_DIR, f"window_{window_id}.npz")
