import geopandas as gpd
import osmnx as ox
import numpy as np
from shapely.geometry import Point, box

wards = gpd.read_file("docs/mumbai_wards.geojson")
g_north = wards[wards["name"] == "G/N"]
boundary_polygon = g_north.geometry.iloc[0]

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
cell_size_deg = 0.0001

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

for i in range(n_rows):
    for j in range(n_cols):
        cell_minx = minx + j * cell_size_deg
        cell_miny = miny + i * cell_size_deg
        cell_center = Point(cell_minx + 0.5 * cell_size_deg, cell_miny + 0.5 * cell_size_deg)
        cell_polygon = box(
            cell_minx,
            cell_miny,
            cell_minx + cell_size_deg,
            cell_miny + cell_size_deg,
        )
        if boundary_polygon.contains(cell_center):
            legal_mask[i, j] = 1

            # Use cell-center classification, consistent with the existing
            # ward-boundary logic and the footfall assignment below.
            if buildings.geometry.intersects(cell_polygon).any():
                building_mask[i, j] = 1

            if road_rows.geometry.intersects(cell_polygon).any():
                road_mask[i, j] = 1

            # Footways and pedestrian paths are kept as possible candidate
            # locations. They are not included in road_mask above.
            if footway_rows.geometry.intersects(cell_polygon).any():
                footway_mask[i, j] = 1

            if water.geometry.intersects(cell_polygon).any():
                water_mask[i, j] = 1

            if railways.geometry.intersects(cell_polygon).any():
                railway_mask[i, j] = 1

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

for _, poi in pois.iterrows():
    col = int((poi.geometry.x - minx) / cell_size_deg)
    row = int((poi.geometry.y - miny) / cell_size_deg)
    if 0 <= row < n_rows and 0 <= col < n_cols:
        footfall_grid[row, col] += 1

max_count = footfall_grid.max()
footfall_grid_normalized = footfall_grid / max_count if max_count > 0 else footfall_grid

np.savez(
    "docs/g_north_data.npz",
    footfall=footfall_grid_normalized,
    legal=legal_mask,
    building=building_mask,
    road=road_mask,
    water=water_mask,
    railway=railway_mask,
    footway=footway_mask,
)
print(f"Grid shape: {footfall_grid_normalized.shape}")
print(f"Legal cells: {int(legal_mask.sum())} / {legal_mask.size}")
print(f"Blocked by buildings: {int(building_mask.sum())}")
print(f"Blocked by roads: {int(road_mask.sum())}")
print(f"Blocked by water: {int(water_mask.sum())}")
print(f"Blocked by railways: {int(railway_mask.sum())}")
print(f"Mapped footway cells: {int(footway_mask.sum())}")
