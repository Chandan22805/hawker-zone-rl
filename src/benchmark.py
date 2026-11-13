"""Benchmark the trained DQN's stall placements against simple baselines.

All policies run through the same HawkerZoneEnv, on the same saved grid, at the
same stall counts, so the numbers are directly comparable. No OSM download is
needed: it only reads the .npz grid and the saved model.

Policies
    Random (legal)    uniform over legal, unoccupied cells (mean of RANDOM_RUNS)
    Footfall-greedy   highest footfall first; ignores spread. Gives the upper
                      bound on total footfall, so "% of footfall bound" <= 100.
    Reward-greedy     at each step picks the cell with the best *immediate* env
                      reward (footfall minus spread term). This is the baseline
                      that shows whether the DQN's long-horizon learning adds
                      anything over a one-step look at the same reward.
    DQN               trained model, invalid cells masked via Q-values.

Run from the project root:
    python src/benchmark.py
    python src/benchmark.py --stalls 100 200 --random-runs 50
"""

import argparse
import csv
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from stable_baselines3 import DQN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hawker_env import HawkerZoneEnv

WARD_NAMES = ["G/N", "F/N", "G/S", "F/S"]
CELL_SIZE_DEG = 0.00025
TRAINED_STALLS = 100
MUMBAI_LAT = 19.1          # only used to convert cell size to metres
LINK_METERS = 400          # stalls within this gap chain into one zone
MIN_ZONE_STALLS = 3        # smaller groups count as isolated stalls

ward_tag = "_".join(w.replace("/", "") for w in WARD_NAMES).lower()
cell_size_label = format(CELL_SIZE_DEG, "g")
DATA_PATH = f"footfall_grid_data/{ward_tag}_data_{cell_size_label}.npz"
MODEL_PATH = os.path.join("models", f"hawker_dqn_{ward_tag}_{cell_size_label}_stalls{TRAINED_STALLS}")

CELL_M_Y = CELL_SIZE_DEG * 111_320
CELL_M_X = CELL_M_Y * np.cos(np.radians(MUMBAI_LAT))


# ---------------------------------------------------------------- policies
def valid_mask(env):
    return (env.legal.ravel() == 1) & (env.occupied.ravel() == 0)


def make_random(rng):
    def pick(env, obs):
        return int(rng.choice(np.flatnonzero(valid_mask(env))))
    return pick


def pick_footfall_greedy(env, obs):
    score = env.footfall.ravel().astype(float)
    score[~valid_mask(env)] = -np.inf
    return int(np.argmax(score))


def pick_reward_greedy(env, obs):
    """Best immediate env reward: footfall - weight * normalized bbox area."""
    idx = np.arange(env.rows * env.cols)
    r, c = np.divmod(idx, env.cols)
    if env.placed_cells:
        pr = np.array([p[0] for p in env.placed_cells])
        pc = np.array([p[1] for p in env.placed_cells])
        spread = (np.maximum(pr.max(), r) - np.minimum(pr.min(), r)) * \
                 (np.maximum(pc.max(), c) - np.minimum(pc.min(), c))
    else:
        spread = np.zeros_like(idx)
    score = env.footfall.ravel() - env.spread_penalty_weight * spread / (env.rows * env.cols)
    score[~valid_mask(env)] = -np.inf
    return int(np.argmax(score))


def make_dqn(model):
    def pick(env, obs):
        q = model.q_net(model.policy.obs_to_tensor(obs)[0]).detach().cpu().numpy()[0]
        q[~valid_mask(env)] = -np.inf
        return int(np.argmax(q))
    return pick


# ----------------------------------------------------------------- metrics
def zone_stats(cells):
    xy = np.array([[c * CELL_M_X, r * CELL_M_Y] for r, c in cells])
    if len(xy) < 2:
        return 0, 0.0
    raw = fcluster(linkage(xy, method="single"), t=LINK_METERS, criterion="distance")
    sizes = np.bincount(raw)
    in_zone = sizes[raw] >= MIN_ZONE_STALLS
    return int((sizes >= MIN_ZONE_STALLS).sum()), float(in_zone.mean())


def run_episode(pick_fn, n_stalls):
    env = HawkerZoneEnv(n_stalls=n_stalls, max_steps=n_stalls)  # masked picks are always valid
    env.load_real_footfall(DATA_PATH)
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(n_stalls):
        obs, reward, terminated, truncated, _ = env.step(pick_fn(env, obs))
        total_reward += reward
        if terminated or truncated:
            break
    cells = env.placed_cells
    rows = [c[0] for c in cells]
    cols = [c[1] for c in cells]
    n_zones, frac_in_zones = zone_stats(cells)
    footfall = float(sum(env.footfall[i, j] for i, j in cells))
    return {
        "placed": len(cells),
        "reward": total_reward,
        "footfall": footfall,
        "bbox_pct": 100 * (max(rows) - min(rows)) * (max(cols) - min(cols)) / (env.rows * env.cols),
        "zones": n_zones,
        "in_zone_pct": 100 * frac_in_zones,
    }


def mean_of(runs):
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}


# -------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stalls", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--random-runs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=os.path.join("visualizations", ward_tag,
                                                      f"cell_size_{cell_size_label}", "results", "benchmark.csv"))
    args = parser.parse_args()

    model = DQN.load(MODEL_PATH, device=args.device)
    rng = np.random.default_rng(0)
    rows_out = []

    for n in args.stalls:
        results = {
            "Random (legal)": mean_of([run_episode(make_random(rng), n) for _ in range(args.random_runs)]),
            "Footfall-greedy": run_episode(pick_footfall_greedy, n),
            "Reward-greedy": run_episode(pick_reward_greedy, n),
            "DQN": run_episode(make_dqn(model), n),
        }
        bound = results["Footfall-greedy"]["footfall"]
        note = "" if n == TRAINED_STALLS else f"  (model trained at {TRAINED_STALLS})"
        print(f"\n=== {n} stalls{note} ===")
        print(f"{'policy':<17}{'reward':>9}{'footfall':>10}{'% bound':>9}{'bbox %':>8}{'zones':>7}{'in-zone %':>11}")
        for name, m in results.items():
            pct = 100 * m["footfall"] / bound if bound else 0.0
            print(f"{name:<17}{m['reward']:>9.2f}{m['footfall']:>10.2f}{pct:>9.1f}"
                  f"{m['bbox_pct']:>8.1f}{m['zones']:>7.1f}{m['in_zone_pct']:>11.1f}")
            rows_out.append({"stalls": n, "policy": name, "footfall_pct_of_bound": pct, **m})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()