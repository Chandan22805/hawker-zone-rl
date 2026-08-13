import geopandas as gpd
import osmnx as ox
import numpy as np
from shapely.geometry import Point

wards = gpd.read_file("docs/mumbai_wards.geojson")
g_north = wards[wards["name"] == "G/N"]
boundary_polygon = g_north.geometry.iloc[0]

pois = ox.features_from_polygon(boundary_polygon, tags={"shop": True, "amenity": True})
pois = pois[pois.geometry.type == "Point"]

minx, miny, maxx, maxy = boundary_polygon.bounds
cell_size_deg = 0.00045

n_cols = int((maxx - minx) / cell_size_deg) + 1
n_rows = int((maxy - miny) / cell_size_deg) + 1
footfall_grid = np.zeros((n_rows, n_cols))
legal_mask = np.zeros((n_rows, n_cols))

for i in range(n_rows):
    for j in range(n_cols):
        cell_center = Point(minx + (j + 0.5) * cell_size_deg, miny + (i + 0.5) * cell_size_deg)
        if boundary_polygon.contains(cell_center):
            legal_mask[i, j] = 1

for _, poi in pois.iterrows():
    col = int((poi.geometry.x - minx) / cell_size_deg)
    row = int((poi.geometry.y - miny) / cell_size_deg)
    if 0 <= row < n_rows and 0 <= col < n_cols:
        footfall_grid[row, col] += 1

max_count = footfall_grid.max()
footfall_grid_normalized = footfall_grid / max_count if max_count > 0 else footfall_grid

np.savez("docs/g_north_data.npz", footfall=footfall_grid_normalized, legal=legal_mask)
print(f"Grid shape: {footfall_grid_normalized.shape}, legal cells: {int(legal_mask.sum())} / {legal_mask.size}")