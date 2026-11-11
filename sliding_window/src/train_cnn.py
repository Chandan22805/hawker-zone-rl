"""Train one DQN window-by-window across the whole city.

The model is always saved to the same file after each window, and finished
windows are recorded in a progress file, so training can be stopped at any
point and resumed later from the next unfinished window.

Windows are trained in the order of config.TRAIN_WINDOWS.

    python sliding_window/src/train_cnn.py                  # all remaining windows
    python sliding_window/src/train_cnn.py --max-windows 2  # train 2, then stop
    python sliding_window/src/train_cnn.py --new-pass       # start another pass over every window
"""

import argparse
import json
import os

from dotenv import load_dotenv
from stable_baselines3 import DQN

import config
from hawker_env import HawkerZoneEnv

parser = argparse.ArgumentParser()
parser.add_argument("--timesteps-per-window", type=int, default=50_000)
parser.add_argument("--max-windows", type=int, default=None, help="Stop after this many windows (resume later)")
parser.add_argument("--new-pass", action="store_true", help="All windows done: start over for another pass, keeping the model")
args = parser.parse_args()

load_dotenv()
device = os.getenv("DEVICE", "cpu").strip().lower()

with open(config.MANIFEST_PATH, encoding="utf-8") as f:
    manifest = json.load(f)
by_id = {w["id"]: w for w in manifest["windows"]}
too_sparse = [w for w in config.TRAIN_WINDOWS if by_id.get(w, {}).get("built") and not by_id[w]["trainable"]]
if too_sparse:
    print(f"Skipping {too_sparse}: fewer than {config.MIN_LEGAL_CELLS} legal cells")
trainable = [w for w in config.TRAIN_WINDOWS if w not in too_sparse]

progress = {"pass": 1, "done": []}
if os.path.exists(config.PROGRESS_PATH):
    with open(config.PROGRESS_PATH, encoding="utf-8") as f:
        progress = json.load(f)
if args.new_pass:
    progress = {"pass": progress["pass"] + 1, "done": []}

remaining = [w for w in trainable if w not in progress["done"]]
if not remaining:
    raise SystemExit(f"Pass {progress['pass']} already covers all {len(trainable)} windows. "
                     "Use --new-pass to train over them again.")
to_train = remaining[:args.max_windows] if args.max_windows else remaining
# Train in order up to the first window that hasn't been built yet.
unbuilt = [w for w in to_train if not by_id.get(w, {}).get("built")]
if unbuilt:
    to_train = to_train[:to_train.index(unbuilt[0])]
    print(f"Not built yet: {unbuilt}. Build them with:\n"
          f"  python sliding_window/src/build_footfall_grid.py --only {' '.join(unbuilt)}")
    if not to_train:
        raise SystemExit("Nothing to train until those are built.")
print(f"Pass {progress['pass']}: {len(progress['done'])}/{len(trainable)} windows done, training {to_train}")

os.makedirs(config.MODELS_DIR, exist_ok=True)
model = None
for window_id in to_train:
    # Training explores unmasked, so many steps are wasted on illegal cells;
    # max_steps headroom lets episodes still reach the later placements.
    env = HawkerZoneEnv(n_stalls=config.N_STALLS, max_steps=config.N_STALLS * 4)
    env.load_real_footfall(config.window_path(window_id))

    if model is None and os.path.exists(config.MODEL_PATH):
        print(f"Resuming from {config.MODEL_PATH}")
        model = DQN.load(config.MODEL_PATH, env=env, device=device)
    elif model is None:
        print("No saved model yet, creating a new one")
        model = DQN(
            "CnnPolicy", env,
            buffer_size=2000, learning_starts=500, batch_size=32,
            # Keep exploring for 30% of each window instead of SB3's default 10%.
            exploration_fraction=0.3,
            # A 256-wide feature layer keeps the 400x400 network about the same
            # size as the earlier 299x319 model, so it still fits in ~4GB VRAM.
            policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=256)),
            verbose=1, device=device,
        )
    else:
        model.set_env(env)

    print(f"--- Window {window_id}: {args.timesteps_per_window} timesteps ---")
    # reset_num_timesteps=True restarts the exploration schedule, so each new
    # window gets explored instead of inheriting the last window's low epsilon.
    model.learn(total_timesteps=args.timesteps_per_window, reset_num_timesteps=True)
    model.save(config.MODEL_PATH)

    progress["done"].append(window_id)
    with open(config.PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    print(f"Saved: {config.MODEL_PATH}  ({len(progress['done'])}/{len(trainable)} windows done)")
