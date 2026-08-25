import os
from dotenv import load_dotenv
from stable_baselines3 import DQN
from hawker_env import HawkerZoneEnv

load_dotenv()
device = os.getenv("DEVICE", "cpu").strip().lower()

env = HawkerZoneEnv(n_stalls=25)
env.load_real_footfall("docs/g_north_data.npz")
model = DQN("CnnPolicy", env, buffer_size=10000, verbose=1, device=device, policy_kwargs=dict(normalize_images=False))
model.learn(total_timesteps=50000)
model.save("hawker_dqn_gnorth")