import torch
import numpy as np
import random

np.random.seed(0)
random.seed(0)

def random_walk(size = 4, seq_len = 32):
    moves = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

    trajectory = []
    pos = np.zeros(2)
    while len(trajectory) < seq_len:
        trajectory.append(pos)
        if random.random() < 0.1:
            move = moves[random.randrange(4)]
        else:
            move = moves[random.randrange(2)]
        pos = pos + move
        pos = np.clip(pos, 0, size - 1)

    trajectory = np.array(trajectory).astype(np.uint8)
    return trajectory.T


class Grid_World_Dataset(torch.utils.data.Dataset):
    def __init__(self, grid_map, feature_map, seq_len = 32, categories = 5):
        super().__init__()
        size = grid_map.shape[0]
        self.trajectory_gen = lambda : random_walk(size, seq_len)
        self.grid_map = grid_map
        self.seq_len = seq_len
        self.reward_map = np.arange(categories).astype(np.float32)
        self.feature_map = feature_map
        #self.encoding = np.eye(categories)

    def __len__(self):
        return 10000000

    def __getitem__(self, index):
        trajectory = self.trajectory_gen()
        blocks = self.grid_map[trajectory[0], trajectory[1]]
        blocks_features = self.feature_map[trajectory[0], trajectory[1]]

        reward = np.sum(self.reward_map[blocks])
        blocks_features = torch.from_numpy(blocks_features).float()

        return blocks_features, reward


