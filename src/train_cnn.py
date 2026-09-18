import os
from dotenv import load_dotenv
from stable_baselines3 import DQN
from hawker_env import HawkerZoneEnv

load_dotenv()
device = os.getenv("DEVICE", "cpu").strip().lower()

WARD_NAMES = ["G/N", "F/N", "G/S"]
CELL_SIZE_DEG = 0.00045
ward_tag = "_".join(w.replace("/", "") for w in WARD_NAMES).lower()
cell_size_label = format(CELL_SIZE_DEG, "g")

env = HawkerZoneEnv(n_stalls=100)
env.load_real_footfall(f"footfall_grid_data/{ward_tag}_data_{cell_size_label}.npz")
model = DQN("CnnPolicy", env, buffer_size=500, learning_starts=500, batch_size=32, verbose=1, device=device, policy_kwargs=dict(normalize_images=False))
model.learn(total_timesteps=50000)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
model_path = os.path.join(MODELS_DIR, f"hawker_dqn_{ward_tag}_{cell_size_label}_stalls{env.n_stalls}")
model.save(model_path)
print(f"Saved: {model_path}.zip")