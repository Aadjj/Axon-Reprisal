import gym
from gym import spaces
import numpy as np
import logging

class ThreatResponseEnv(gym.Env):
    def __init__(self, state_dim=7):
        super(ThreatResponseEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        self.state = np.zeros(state_dim)
        self.action_map = {0: "block_ip", 1: "isolate_host", 2: "quarantine_file", 3: "no_action"}

    def reset(self, initial_state=None):
        if initial_state is not None:
            self.state = initial_state.astype(np.float32)
        else:
            self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
        return self.state

    def step(self, action):
        reward = self._calculate_reward(action)
        done = True
        info = {"action_taken": self.action_map[action]}
        return self.state, reward, done, info

    def _calculate_reward(self, action):
        if action == 0: return 5
        if action == 1: return 10
        if action == 2: return 3
        return -5