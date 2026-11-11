"""DBSCAN: density-based clustering.

A zone grows from "core" stalls that have at least --min-samples stalls
within --eps metres. Stalls in sparse areas are labelled noise rather than
forced into a zone.

Run from the project root:
    python src/clustering/dbscan/cluster_dbscan.py
"""

import argparse
import os
import sys

from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import common

parser = argparse.ArgumentParser()
parser.add_argument("--eps", type=float, default=400, help="Neighbourhood radius in metres")
parser.add_argument("--min-samples", type=int, default=3, help="Stalls needed within eps to be a core stall")
args = parser.parse_args()


def cluster(xy):
    return DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(xy)


common.run(cluster, f"DBSCAN (eps={args.eps:g}m, min_samples={args.min_samples})", os.path.dirname(os.path.abspath(__file__)))
