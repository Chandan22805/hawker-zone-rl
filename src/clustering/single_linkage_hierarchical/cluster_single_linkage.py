"""Single-linkage hierarchical clustering ("friends-of-friends").

Stalls chain into one zone if each is within --link-meters of the next.
Finds any number of zones of any shape, but long thin chains can merge
stalls whose ends are far apart.

Run from the project root:
    python src/clustering/single_linkage_hierarchical/cluster_single_linkage.py
"""

import argparse
import os
import sys

from scipy.cluster.hierarchy import fcluster, linkage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import common

parser = argparse.ArgumentParser()
parser.add_argument("--link-meters", type=float, default=400, help="Max gap between neighbouring stalls in a zone")
args = parser.parse_args()


def cluster(xy):
    return fcluster(linkage(xy, method="single"), t=args.link_meters, criterion="distance")


common.run(cluster, f"Single linkage ({args.link_meters:g}m link)", os.path.dirname(os.path.abspath(__file__)))
