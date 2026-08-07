from stable_baselines3 import DQN
from src.hawker_env import HawkerZoneEnv

env = HawkerZoneEnv()
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("hawker_dqn")