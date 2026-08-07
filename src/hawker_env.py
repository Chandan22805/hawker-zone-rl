import numpy as np
import gymnasium as gym
from gymnasium import spaces

class HawkerZoneEnv(gym.Env):
    def __init__(self, grid_size=10, n_stalls=5, illegal_frac=0.2):
        super().__init__()
        self.grid_size = grid_size
        self.n_stalls = n_stalls
        self.illegal_frac = illegal_frac

        # action = pick a flattened grid cell index
        self.action_space = spaces.Discrete(grid_size * grid_size)

        # observation = footfall, legal flag, occupied flag per cell, flattened + stall progress
        obs_len = grid_size * grid_size * 3 + 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_len,), dtype=np.float32)

        self.reset()

    def _generate_ward(self):
        self.footfall = np.random.rand(self.grid_size, self.grid_size)

        # cell_type: 0 = empty space, 1 = building, 2 = road, 3 = illegal (no-hawking) zone
        self.cell_type = np.zeros((self.grid_size, self.grid_size), dtype=int)

        n_cells = self.grid_size * self.grid_size
        n_building = int(0.25 * n_cells)
        n_road = int(0.15 * n_cells)
        n_illegal = int(self.illegal_frac * n_cells)

        idx = np.random.permutation(n_cells)
        self.cell_type.flat[idx[:n_building]] = 1
        self.cell_type.flat[idx[n_building:n_building+n_road]] = 2
        self.cell_type.flat[idx[n_building+n_road:n_building+n_road+n_illegal]] = 3

        # legal = only plain empty cells (type 0) can host a stall
        self.legal = (self.cell_type == 0).astype(np.float32)
        self.occupied = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
    def _get_obs(self):
        stall_progress = np.array([self.current_stall / self.n_stalls], dtype=np.float32)
        return np.concatenate([
            self.footfall.flatten(),
            self.legal.flatten(),
            self.occupied.flatten(),
            stall_progress
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_ward()
        self.current_stall = 0
        self.placed_cells = []
        self.steps_taken = 0
        self.max_steps = 50   # hard cap regardless of stalls placed
        return self._get_obs(), {}

    def step(self, action):
        i, j = divmod(action, self.grid_size)
        reward = 0.0
        terminated = False
        truncated = False
        self.steps_taken += 1

        if self.legal[i, j] == 0 or self.occupied[i, j] == 1:
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

            reward = footfall_score - 0.05 * spread_penalty
            self.current_stall += 1

        if self.current_stall >= self.n_stalls:
            terminated = True
        if self.steps_taken >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}