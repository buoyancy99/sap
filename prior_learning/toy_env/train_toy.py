import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from prior_learning.toy_env.toyloader import toyenv_Dataset
size = 8
seq_len = 32
categories = 16
batch_size = 128
feature_dim = 16
features = np.random.random((categories, feature_dim))

train_loader = DataLoader(toyenv_Dataset(features, size, seq_len, categories), batch_size = batch_size, num_workers= 40, shuffle = True)
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

    blocks, masks, rewards = [d.cuda() for d in data]
    blocks = blocks.view(batch_size * seq_len, feature_dim)
    rewards_hat = net(blocks)
    rewards_hat = rewards_hat.view(batch_size, seq_len)
    reg = torch.mean(torch.abs(rewards_hat)) * 0.01
    rewards_hat = torch.sum(rewards_hat * masks, 1)
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
        result = net(torch.from_numpy(features).float().cuda()).flatten().detach().cpu().numpy()
        print('=' * 40)
        print(result)
        print('='*40)




