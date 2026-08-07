from src.hawker_env import HawkerZoneEnv

env = HawkerZoneEnv()
obs, info = env.reset()
print("Initial obs shape:", obs.shape)

for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: action={action}, reward={reward:.3f}, terminated={terminated}")