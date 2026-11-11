"""Complete-linkage hierarchical clustering.

Every pair of stalls in a zone is within --max-diameter of each other, so
zones are compact and can't stretch through a thin chain of stalls the way
single linkage can.

Run from the project root:
    python src/clustering/complete_linkage_hierarchical/cluster_complete_linkage.py
"""

import argparse
import os
import sys

from scipy.cluster.hierarchy import fcluster, linkage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import common

parser = argparse.ArgumentParser()
parser.add_argument("--max-diameter", type=float, default=800, help="Max distance between any two stalls in a zone")
args = parser.parse_args()


def cluster(xy):
    return fcluster(linkage(xy, method="complete"), t=args.max_diameter, criterion="distance")


common.run(cluster, f"Complete linkage ({args.max_diameter:g}m diameter)", os.path.dirname(os.path.abspath(__file__)))
