import torch
import numpy as np
import random

np.random.seed(0)
random.seed(0)

def random_walk(size = 4, seq_len = 32):
    destination = np.ones((2, 2)) * (size - 1)
    moves = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

    trajectory = []
    pos = np.zeros(2)
    while not np.equal(pos, destination).all() and len(trajectory) < seq_len:
        trajectory.append(pos)
        if random.random() < 0.1:
            move = moves[random.randrange(4)]
        else:
            move = moves[random.randrange(2)]
        pos = pos + move
        pos = np.clip(pos, 0, size - 1)

    trajectory = np.array(trajectory).astype(np.uint8)
    return trajectory.T


class toyenv_Dataset(torch.utils.data.Dataset):
    def __init__(self, features, size = 4 ,seq_len = 32, categories = 5):
        super().__init__()
        self.trajectory_gen = lambda : random_walk(size, seq_len)
        self.grid_world = np.random.randint(0, categories, size=(size, size))
        self.seq_len = seq_len
        self.reward_map = np.arange(categories).astype(np.float32)
        self.feature_dim = features.shape[-1]
        noise = np.random.normal(scale = 0.05, size=(size, size, self.feature_dim))
        self.encoding_map = features[self.grid_world].reshape((size, size, self.feature_dim)) + noise
        #self.encoding = np.eye(categories)


    def __len__(self):
        return 10000000

    def __getitem__(self, index):
        trajectory = self.trajectory_gen()
        blocks = self.grid_world[trajectory[0], trajectory[1]]
        blocks_features = self.encoding_map[trajectory[0], trajectory[1]]
        pad = self.seq_len - len(blocks)
        mask = np.ones(self.seq_len)

        if pad > 0:
            padded_blocks = np.zeros((self.seq_len), dtype = np.uint8)
            padded_blocks[: len(blocks)] = blocks
            padded_features = np.zeros((self.seq_len, self.feature_dim))
            padded_features[: len(blocks)] = blocks_features
            mask[len(blocks):] = 0
            blocks = padded_blocks
            blocks_features = padded_features

        reward = np.sum(self.reward_map[blocks])
        blocks_features = torch.from_numpy(blocks_features).float()
        mask = torch.from_numpy(mask).float()

        return blocks_features, mask, reward


