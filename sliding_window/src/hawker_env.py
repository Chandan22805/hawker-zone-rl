import numpy as np
import gymnasium as gym
from gymnasium import spaces


class HawkerZoneEnv(gym.Env):
    def __init__(self, n_stalls=15, max_steps=100, spread_penalty_weight=1.0):
        super().__init__()
        self.n_stalls = n_stalls
        self.max_steps = max_steps
        # Weight applied to the *normalized* spread penalty (bounding-box area
        # divided by total grid area, so it's always in a 0-1 range like
        # footfall_score, regardless of grid size).
        self.spread_penalty_weight = spread_penalty_weight
        self._loaded = False

    def load_real_footfall(self, npz_path):
        data = np.load(npz_path)
        self.footfall = data["footfall"]
        self.legal = data["legal"]
        self.rows, self.cols = self.footfall.shape
        self.action_space = spaces.Discrete(self.rows * self.cols)
        # uint8 observations: at 400x400 a float32 obs is ~1.9MB, which makes
        # the replay buffer several GB. uint8 cuts that 4x; SB3's CnnPolicy
        # rescales it back to 0-1 internally.
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, self.rows, self.cols), dtype=np.uint8)
        self._footfall_u8 = np.round(self.footfall * 255).astype(np.uint8)
        self._legal_u8 = (self.legal * 255).astype(np.uint8)
        self._loaded = True

    def _get_obs(self):
        return np.stack([self._footfall_u8, self._legal_u8, (self.occupied * 255).astype(np.uint8)], axis=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        assert self._loaded, "call load_real_footfall() before reset()"
        self.occupied = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.current_stall = 0
        self.placed_cells = []
        self.steps_taken = 0
        return self._get_obs(), {}

    def step(self, action):
        i, j = divmod(int(action), self.cols)
        reward, terminated, truncated = 0.0, False, False
        self.steps_taken += 1

        if self.occupied[i, j] == 1 or self.legal[i, j] == 0:
            reward = -1.0
        else:
            self.occupied[i, j] = 1
            self.placed_cells.append((i, j))
            footfall_score = self.footfall[i, j]

            if len(self.placed_cells) > 1:
                rows = [c[0] for c in self.placed_cells]
                cols = [c[1] for c in self.placed_cells]
                spread_penalty = (max(rows) - min(rows)) * (max(cols) - min(cols))
            else:
                spread_penalty = 0

            spread_penalty_ratio = spread_penalty / (self.rows * self.cols)
            reward = footfall_score - self.spread_penalty_weight * spread_penalty_ratio
            self.current_stall += 1

        if self.current_stall >= self.n_stalls:
            terminated = True
        if self.steps_taken >= self.max_steps:
            truncated = True
        return self._get_obs(), reward, terminated, truncated, {}
