from stable_baselines3 import DQN
from hawker_env import HawkerZoneEnv

print("Creating environment...")
env = HawkerZoneEnv(n_stalls=15)

print("Loading real data...")
env.load_real_data("g_north_footfall_grid.npy")
print(f"Grid shape: {env.footfall.shape}, obs space: {env.observation_space.shape}")

print("Testing reset...")
obs, info = env.reset()
print(f"Obs shape from reset: {obs.shape}")

print("Creating DQN model with CnnPolicy...")
model = DQN(
    "CnnPolicy",
    env,
    buffer_size=10000,
    verbose=1,
    policy_kwargs=dict(normalize_images=False)
)
print("Model created successfully.")

print("Starting training...")
model.learn(total_timesteps=50000)
print("Training complete.")

model.save("hawker_dqn_cnn")
print("Saved.")