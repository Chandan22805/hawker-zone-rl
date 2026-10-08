import os
from dotenv import load_dotenv
from stable_baselines3 import DQN
from hawker_env import HawkerZoneEnv

load_dotenv()
device = os.getenv("DEVICE", "cpu").strip().lower()

WARD_NAMES = ["G/N", "F/N", "G/S", "F/S"]
CELL_SIZE_DEG = 0.00025
ward_tag = "_".join(w.replace("/", "") for w in WARD_NAMES).lower()
cell_size_label = format(CELL_SIZE_DEG, "g")

N_STALLS = 100
# Training explores with unmasked epsilon-greedy actions, so it frequently
# tries illegal/occupied cells while exploring — unlike inference, those
# wasted attempts still count against max_steps. Give real headroom above
# n_stalls so episodes can actually reach deep into placement counts instead
# of truncating early almost every time.
env = HawkerZoneEnv(n_stalls=N_STALLS, max_steps=N_STALLS * 4)
env.load_real_footfall(f"footfall_grid_data/{ward_tag}_data_{cell_size_label}.npz")
model = DQN("CnnPolicy", env, buffer_size=2000, learning_starts=500, batch_size=32, verbose=1, device=device, policy_kwargs=dict(normalize_images=False))
model.learn(total_timesteps=100000)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
model_path = os.path.join(MODELS_DIR, f"hawker_dqn_{ward_tag}_{cell_size_label}_stalls{env.n_stalls}")
model.save(model_path)
print(f"Saved: {model_path}.zip")