"""HDBSCAN: hierarchical density-based clustering.

Like DBSCAN but with no fixed radius: it finds zones at whatever density
they naturally occur, so a tight market street and a looser neighbourhood
cluster can both be detected in the same run.

Run from the project root:
    python src/clustering/hdbscan/cluster_hdbscan.py
"""

import argparse
import os
import sys

from sklearn.cluster import HDBSCAN

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import common

parser = argparse.ArgumentParser()
parser.add_argument("--min-cluster-size", type=int, default=3, help="Smallest group counted as a zone")
args = parser.parse_args()


def cluster(xy):
    return HDBSCAN(min_cluster_size=args.min_cluster_size).fit_predict(xy)


common.run(cluster, f"HDBSCAN (min_cluster_size={args.min_cluster_size})", os.path.dirname(os.path.abspath(__file__)))
