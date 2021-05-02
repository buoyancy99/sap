import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from prior_learning.gridworld.dataloader import Grid_World_Dataset
import numpy as np

batch_size = 128

def get_maps(grid_map, feature_map, feature_encoding, seq_len, categories):
    feature_dim = feature_map.shape[-1]
    size = grid_map.shape[0]
    # noise = np.random.normal(scale=0.05, size=(size, size, feature_dim))
    # feature_map = feature_encoding[grid_map.flatten()].reshape((size, size, feature_dim)) + noise


    train_loader = DataLoader(Grid_World_Dataset(grid_map, feature_map, seq_len, categories), batch_size=batch_size, num_workers=40, shuffle=True)
    net = nn.Sequential(
        nn.Linear(feature_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

    net.cuda()
    criteria = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    reg_sum = 0
    loss_sum = 0

    for i, data in enumerate(train_loader):
        if i == 29999:
            optimizer = optim.Adam(net.parameters(), lr=1e-4)

        blocks, rewards = [d.cuda() for d in data]
        blocks = blocks.view(batch_size * seq_len, feature_dim)
        rewards_hat = net(blocks)
        rewards_hat = rewards_hat.view(batch_size, seq_len)
        reg = torch.mean(torch.abs(rewards_hat)) * 0.005
        rewards_hat = torch.sum(rewards_hat, 1)
        loss = criteria(rewards_hat, rewards) + reg

        loss_sum += loss.item()
        reg_sum += reg.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 2000 == 1999:
            print('[{}] loss: {}, reg: {}'.format(i + 1, loss_sum / 100, reg_sum / 100))
            loss_sum = 0
            reg_sum = 0

        if i % 10000 == 9999:
            predicted_reward = net(torch.from_numpy(feature_encoding).float().cuda()).flatten().detach().cpu().numpy()
            print('=' * 40)
            print(predicted_reward)
            print('=' * 40)

        if i >= 40000:
            break

    reward_map = net(torch.from_numpy(feature_map.reshape((size * size, feature_dim))).float().cuda()).detach().cpu().numpy().reshape((size,  size))

    return reward_map, net

if __name__ == "__main__":
    import os

    size = 8
    categories = 16
    feature_dim = 16
    grid_map = np.random.randint(0, categories, size=(size, size))
    feature_encoding = np.random.random((categories, feature_dim))

    noise = np.random.normal(scale=0.05, size=(size, size, feature_dim))
    feature_map = feature_encoding[grid_map.flatten()].reshape((size, size, feature_dim)) + noise

    reward_map, net = get_maps(grid_map, feature_map, feature_encoding, 32, 16)
    # torch.save(net.cpu().state_dict(), os.path.join('prior_learning', 'gridworld', 'ckpts', 'reward.model'))
