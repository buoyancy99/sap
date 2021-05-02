import gym
import numpy as np
from gym import spaces

class Grid_World_Env(gym.Env):
    def __init__(self, reward_map, reward_map_hat):
        super().__init__()
        self.size = reward_map.shape[0]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.size), spaces.Discrete(self.size)))
        self.reward_map = reward_map
        self.reward_map_hat = reward_map_hat
        self.moves = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=int)
        self.reset()

    def reset(self):
        self.total_steps = 0
        self.total_reward = 0
        self.pos = np.zeros(2, dtype=int)
        return (self.pos[0], self.pos[1])

    def step(self, action):
        move = self.moves[action]
        self.pos = self.pos + move
        self.pos = np.clip(self.pos, 0, self.size - 1)
        obs = (self.pos[0], self.pos[1])
        reward = self.reward_map[self.pos[0], self.pos[1]]
        reward_hat = self.reward_map_hat[self.pos[0], self.pos[1]]
        done = self.total_steps >= 32
        info = {'total_steps': self.total_steps,
                'total_reward': self.total_reward,
                'predicted_reward': reward_hat
                }

        self.total_steps += 1
        self.total_reward += reward

        return obs, reward, done, info

    def backup(self):
        self.total_steps_backup = self.total_steps
        self.pos_backup = self.pos
        self.total_reward_backup = self.total_reward

    def restore(self):
        self.total_steps = self.total_steps_backup
        self.pos = self.pos_backup
        self.total_reward = self.total_reward_backup
