# Grid, Cell Size, and Data Guide

This document explains how the geographic area becomes a grid and how that grid is used by the reinforcement-learning environment.

---

## 1. The geographic area

`build_footfall_grid.py` loads Mumbai ward boundaries from:

```text
docs/mumbai_wards.geojson
```

It selects the ward named `G/N` and gets its geographic bounding box:

```python
minx, miny, maxx, maxy = boundary_polygon.bounds
```

Here:

- `x` means longitude;
- `y` means latitude.

The bounding box is a rectangle around the ward. Because the actual ward boundary is irregular, some cells in this rectangle fall outside the ward.

---

## 2. Cell size

The grid cell size is defined here:

```python
cell_size_deg = 0.00045
```

This means each cell is approximately `0.00045` degrees wide and `0.00045` degrees tall.

Around Mumbai, this is approximately 50 metres by 50 metres. The exact physical size varies slightly because degrees of longitude and latitude do not represent exactly the same distance everywhere.

---

## 3. Number of rows and columns

The number of columns and rows is calculated from the bounding box dimensions:

```python
n_cols = int((maxx - minx) / cell_size_deg) + 1
n_rows = int((maxy - miny) / cell_size_deg) + 1
```

Therefore:

- more geographic area produces more rows or columns;
- a smaller cell size produces more rows or columns;
- a larger cell size produces fewer rows or columns.

The current saved data has:

```text
Rows:            100
Columns:          90
Total cells:   9,000
```

The arrays use NumPy shape `(rows, columns)`, so the current shape is:

```text
(100, 90)
```

---

## 4. Row and column locations

For each row `i` and column `j`, the script calculates the cell center:

```python
cell_center = Point(
    minx + (j + 0.5) * cell_size_deg,
    miny + (i + 0.5) * cell_size_deg
)
```

The coordinate system works as follows:

- increasing `i` moves northward;
- increasing `j` moves eastward;
- `(0, 0)` is near the southwest corner of the bounding box.

---

## 5. Legal and illegal cells

The script creates a legal mask with the same shape as the grid:

```python
legal_mask = np.zeros((n_rows, n_cols))
```

It checks whether each cell center is inside the actual ward boundary:

```python
if boundary_polygon.contains(cell_center):
    legal_mask[i, j] = 1
```

The values mean:

```text
1 = inside the ward / legal according to this project
0 = outside the ward / illegal according to this project
```

The current data contains:

```text
Legal cells:    3,711
Illegal cells:  5,289
```

This legal flag only represents whether a cell is inside the selected ward. It does not currently represent every real-world hawker regulation, road clearance, pedestrian-width rule, or restricted zone yet.

---

## 6. Footfall values

The script queries OpenStreetMap for shops and amenities:

```python
tags={"shop": True, "amenity": True}
```

For each returned point, it calculates its grid cell:

```python
col = int((poi.geometry.x - minx) / cell_size_deg)
row = int((poi.geometry.y - miny) / cell_size_deg)
```

Then it increments that cell's count:

```python
footfall_grid[row, col] += 1
```

The count is normalized using the largest cell count:

```python
footfall_grid_normalized = footfall_grid / max_count
```

The final footfall values range from `0.0` to `1.0`.

Important: this is an estimated footfall score based on mapped shops and amenities. It is not a direct measurement of pedestrian traffic.

---

## 7. Synchronization of data

The footfall grid and legal mask are created with the same dimensions:

```python
footfall_grid = np.zeros((n_rows, n_cols))
legal_mask = np.zeros((n_rows, n_cols))
```

They use the same row and column coordinates. Therefore:

```python
footfall[i, j]
legal[i, j]
```

always refer to the same geographic cell.

The saved file contains both arrays:

```text
docs/g_north_data.npz
```

Its contents are:

```python
footfall = data["footfall"]
legal = data["legal"]
```

---

## 8. How the environment loads the grid

In `hawker_env.py`:

```python
self.footfall = data["footfall"]
self.legal = data["legal"]
self.rows, self.cols = self.footfall.shape
```

The environment does not independently calculate the rows, columns, or cell size. It reads the shape of the saved array.

For the current data:

```text
self.rows = 100
self.cols = 90
```

---

## 9. How cells become actions

The environment creates one action for every grid cell:

```python
n_cells = self.rows * self.cols
self.action_space = spaces.Discrete(n_cells)
```

With the current grid:

```text
100 × 90 = 9,000 possible actions
```

The DQN outputs a single action number. The environment converts that number into a row and column:

```python
i, j = divmod(action, self.cols)
```

Examples:

```text
Action 0     → row 0, column 0
Action 1     → row 0, column 1
Action 89    → row 0, column 89
Action 90    → row 1, column 0
Action 8999  → row 99, column 89
```

---

## 10. Occupancy during an episode

The environment creates a third grid at runtime:

```python
self.occupied = np.zeros((self.rows, self.cols))
```

Initially every cell is available. When the agent places a stall:

```python
self.occupied[i, j] = 1
```

The observation given to the model contains three aligned layers:

```text
Layer 1: footfall
Layer 2: legal
Layer 3: occupied
```

For example:

```text
footfall[42, 17] = 0.8
legal[42, 17]    = 1
occupied[42, 17] = 0
```

This means the cell has relatively high estimated footfall, is legal according to the current mask, and is still available.

After placement, `occupied[42, 17]` becomes `1`, so the cell cannot be selected again in that episode.

---

## Summary

```text
Ward boundary
    ↓
Bounding box
    ↓
0.00045-degree cells
    ↓
100 rows × 90 columns
    ↓
Footfall and legal arrays aligned cell-by-cell
    ↓
9,000 possible DQN actions
```

The grid size is fixed when `g_north_data.npz` is generated. The footfall and legal values remain fixed during an episode. Only the occupancy layer changes as the agent places stalls.
