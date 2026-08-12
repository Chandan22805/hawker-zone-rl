from src.hawker_env import HawkerZoneEnv

env = HawkerZoneEnv()
env.load_real_footfall("g_north_footfall_grid.npy")
obs, info = env.reset()
print("Initial obs shape:", obs.shape)

for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: action={action}, reward={reward:.3f}, terminated={terminated}")