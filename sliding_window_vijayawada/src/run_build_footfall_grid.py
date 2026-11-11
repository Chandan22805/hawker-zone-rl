"""Build Vijayawada's 400x400 windows (cells outside the city are illegal).

Runs sliding_window/src/build_footfall_grid.py with Vijayawada's config. Takes the same
flags as that script. Run from the project root:

    python sliding_window_vijayawada/src/run_build_footfall_grid.py
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.normpath(os.path.join(HERE, "..", "..", "sliding_window", "src"))
# HERE stays first on the path, so the shared code's `import config` loads Vijayawada's config.
sys.path.insert(1, SHARED)
runpy.run_path(os.path.join(SHARED, "build_footfall_grid.py"), run_name="__main__")
