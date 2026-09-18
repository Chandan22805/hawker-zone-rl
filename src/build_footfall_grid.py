import os
import geopandas as gpd
import osmnx as ox
import numpy as np
import shapely
from shapely.ops import unary_union
from shapely.geometry import box

# G/N is combined with its longest-bordering neighbor, F/N, into one
# contiguous study area for grid generation.
WARD_NAMES = ["G/N", "F/N", "G/S"]
ward_tag = "_".join(w.replace("/", "") for w in WARD_NAMES).lower()

wards = gpd.read_file("docs/mumbai_wards.geojson")
selected_wards = wards[wards["name"].isin(WARD_NAMES)]
boundary_polygon = unary_union(selected_wards.geometry)

pois = ox.features_from_polygon(boundary_polygon, tags={"shop": True, "amenity": True})
pois = pois[pois.geometry.type == "Point"]

# Physical obstacles. These are deliberately kept separate so that the
# generated data can later explain why a cell was marked illegal.
buildings = ox.features_from_polygon(boundary_polygon, tags={"building": True})
roads = ox.features_from_polygon(boundary_polygon, tags={"highway": True})
roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]
water = ox.features_from_polygon(
    boundary_polygon,
    tags={"natural": ["water", "wetland"], "water": True},
)
railways = ox.features_from_polygon(boundary_polygon, tags={"railway": True})

minx, miny, maxx, maxy = boundary_polygon.bounds
cell_size_deg = 0.00045

cell_size_label = format(cell_size_deg, "g")
OUTPUT_DIR = "footfall_grid_data"
OUTPUT_PATH = f"{OUTPUT_DIR}/{ward_tag}_data_{cell_size_label}.npz"

n_cols = int((maxx - minx) / cell_size_deg) + 1
n_rows = int((maxy - miny) / cell_size_deg) + 1
footfall_grid = np.zeros((n_rows, n_cols))
legal_mask = np.zeros((n_rows, n_cols))
building_mask = np.zeros((n_rows, n_cols))
road_mask = np.zeros((n_rows, n_cols))
water_mask = np.zeros((n_rows, n_cols))
railway_mask = np.zeros((n_rows, n_cols))
footway_mask = np.zeros((n_rows, n_cols))

# Footways, paths, pedestrian areas, and cycleways are not blocked here:
# they may be possible vending locations, subject to future legal rules.
blocked_road_types = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "unclassified", "residential", "service",
    "living_street",
}
footway_types = {"footway", "path", "pedestrian", "cycleway", "steps", "corridor"}
road_rows = roads[roads["highway"].apply(
    lambda value: any(
        road_type in blocked_road_types
        for road_type in (value if isinstance(value, list) else [value])
    )
)]
footway_rows = roads[roads["highway"].apply(
    lambda value: any(
        road_type in footway_types
        for road_type in (value if isinstance(value, list) else [value])
    )
)]

# --- Vectorized grid classification -----------------------------------
# Build every cell center in one shot and test containment with a single
# vectorized shapely call instead of a Python-level double loop.
row_idx, col_idx = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
row_idx = row_idx.ravel()
col_idx = col_idx.ravel()

center_x = minx + (col_idx + 0.5) * cell_size_deg
center_y = miny + (row_idx + 0.5) * cell_size_deg
centers = shapely.points(center_x, center_y)

inside = shapely.contains(boundary_polygon, centers)
legal_mask[row_idx[inside], col_idx[inside]] = 1

# Only build/test cell polygons for cells that are actually inside the
# boundary (mirrors the original behavior: outside cells stay 0 everywhere).
legal_row_idx = row_idx[inside]
legal_col_idx = col_idx[inside]
legal_minx = minx + legal_col_idx * cell_size_deg
legal_miny = miny + legal_row_idx * cell_size_deg
cell_polygons = shapely.box(
    legal_minx, legal_miny,
    legal_minx + cell_size_deg, legal_miny + cell_size_deg,
)
cells_gdf = gpd.GeoDataFrame(
    {"row": legal_row_idx, "col": legal_col_idx},
    geometry=cell_polygons,
    crs=wards.crs,
)


def mark_intersecting_cells(mask, layer_gdf):
    """Spatial-join (STRtree-indexed) instead of per-cell .intersects().any()."""
    if layer_gdf.empty:
        return
    hits = gpd.sjoin(cells_gdf, layer_gdf[["geometry"]], predicate="intersects", how="inner")
    mask[hits["row"].to_numpy(), hits["col"].to_numpy()] = 1


mark_intersecting_cells(building_mask, buildings)
mark_intersecting_cells(road_mask, road_rows)
# Footways and pedestrian paths are kept as possible candidate locations.
# They are not included in road_mask above.
mark_intersecting_cells(footway_mask, footway_rows)
mark_intersecting_cells(water_mask, water)
mark_intersecting_cells(railway_mask, railways)

legal_mask[
    (building_mask == 1)
    | (road_mask == 1)
    | (water_mask == 1)
    | (railway_mask == 1)
] = 0

# Explicitly preserve mapped footway/pedestrian cells as eligible, unless
# another physical obstacle has already blocked them.
legal_mask[
    (footway_mask == 1)
    & (road_mask == 0)
    & (building_mask == 0)
    & (water_mask == 0)
    & (railway_mask == 0)
] = 1

# --- Vectorized footfall counting ---------------------------------------
poi_x = pois.geometry.x.to_numpy()
poi_y = pois.geometry.y.to_numpy()
poi_col = ((poi_x - minx) / cell_size_deg).astype(int)
poi_row = ((poi_y - miny) / cell_size_deg).astype(int)
valid = (poi_row >= 0) & (poi_row < n_rows) & (poi_col >= 0) & (poi_col < n_cols)
np.add.at(footfall_grid, (poi_row[valid], poi_col[valid]), 1)

max_count = footfall_grid.max()
footfall_grid_normalized = footfall_grid / max_count if max_count > 0 else footfall_grid

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez(
    OUTPUT_PATH,
    footfall=footfall_grid_normalized,
    legal=legal_mask,
    building=building_mask,
    road=road_mask,
    water=water_mask,
    railway=railway_mask,
    footway=footway_mask,
)
print(f"Saved: {OUTPUT_PATH}")
print(f"Grid shape: {footfall_grid_normalized.shape}")
print(f"Legal cells: {int(legal_mask.sum())} / {legal_mask.size}")
print(f"Blocked by buildings: {int(building_mask.sum())}")
print(f"Blocked by roads: {int(road_mask.sum())}")
print(f"Blocked by water: {int(water_mask.sum())}")
print(f"Blocked by railways: {int(railway_mask.sum())}")
print(f"Mapped footway cells: {int(footway_mask.sum())}")
