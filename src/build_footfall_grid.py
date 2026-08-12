import geopandas as gpd
import osmnx as ox
import numpy as np

wards = gpd.read_file("docs/mumbai_wards.geojson")
g_north = wards[wards["name"] == "G/N"]
print(f"Found {len(g_north)} matching ward(s)")

boundary_polygon = g_north.geometry.iloc[0]

pois = ox.features_from_polygon(boundary_polygon, tags={"shop": True, "amenity": True})
pois = pois[pois.geometry.type == "Point"]
print(f"POIs found: {len(pois)}")

minx, miny, maxx, maxy = boundary_polygon.bounds
cell_size_deg = 0.00045  # ~50m

n_cols = int((maxx - minx) / cell_size_deg) + 1
n_rows = int((maxy - miny) / cell_size_deg) + 1
footfall_grid = np.zeros((n_rows, n_cols))

for _, poi in pois.iterrows():
    col = int((poi.geometry.x - minx) / cell_size_deg)
    row = int((poi.geometry.y - miny) / cell_size_deg)
    if 0 <= row < n_rows and 0 <= col < n_cols:
        footfall_grid[row, col] += 1

max_count = footfall_grid.max()
footfall_grid_normalized = footfall_grid / max_count if max_count > 0 else footfall_grid
np.save("g_north_footfall_grid.npy", footfall_grid_normalized)
print(f"Grid shape: {footfall_grid_normalized.shape}, max in one cell: {int(max_count)}")