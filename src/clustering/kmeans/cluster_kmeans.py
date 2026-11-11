"""K-Means clustering.

Splits stalls into exactly --k zones around k centres. Every stall is
assigned to some zone (no noise), so it shows where k zone centres would sit,
but distant stalls get pulled into zones they aren't really part of.

Run from the project root:
    python src/clustering/kmeans/cluster_kmeans.py
"""

import argparse
import os
import sys

from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import common

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=12, help="Number of zones")
args = parser.parse_args()


def cluster(xy):
    return KMeans(n_clusters=args.k, n_init=10, random_state=0).fit_predict(xy)


common.run(cluster, f"K-Means (k={args.k})", os.path.dirname(os.path.abspath(__file__)))
