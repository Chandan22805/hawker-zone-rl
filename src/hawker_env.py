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
        # footfall_score, regardless of grid size). Keeping both terms on the
        # same scale is what makes the reward comparable across differently
        # sized cities/grids.
        self.spread_penalty_weight = spread_penalty_weight
        self._loaded = False

    def load_real_footfall(self, npz_path):
        data = np.load(npz_path)
        self.footfall = data["footfall"]
        self.legal = data["legal"]
        self.rows, self.cols = self.footfall.shape
        n_cells = self.rows * self.cols
        self.action_space = spaces.Discrete(n_cells)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.rows, self.cols), dtype=np.float32)
        self._loaded = True

    def _get_obs(self):
        return np.stack([self.footfall, self.legal, self.occupied], axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        assert self._loaded, "call load_real_data() before reset()"
        self.occupied = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.current_stall = 0
        self.placed_cells = []
        self.steps_taken = 0
        return self._get_obs(), {}

    def step(self, action):
        i, j = divmod(action, self.cols)
        reward, terminated, truncated = 0.0, False, False
        self.steps_taken += 1

        if self.occupied[i, j] == 1 or self.legal[i,j]==0:
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

            # Normalize by total grid area so this stays in a 0-1 range
            # comparable to footfall_score, regardless of how big the grid is.
            spread_penalty_ratio = spread_penalty / (self.rows * self.cols)

            reward = footfall_score - self.spread_penalty_weight * spread_penalty_ratio
            self.current_stall += 1

        if self.current_stall >= self.n_stalls:
            terminated = True
        if self.steps_taken >= self.max_steps:
            truncated = True
        return self._get_obs(), reward, terminated, truncated, {}