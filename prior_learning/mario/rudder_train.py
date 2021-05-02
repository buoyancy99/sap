import torch
import torch.optim as optim
import torch.nn as nn
from config.mario_config import config

from prior_learning.mario.rudder_dataloader import Rudder_Mario_Dataset
from prior_learning.mario.rudder_model import Rudder_Net

trained_on = config["trained_on"]

BATCH_SIZE = 256
NUM_WORKERS = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.005
save_dir =  "prior_learning/mario/ckpts/rudder_reward_{}.model".format(trained_on)

def train_rudder_mario():
    dataset = Rudder_Mario_Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    rudder_net = Rudder_Net(64).cuda()
    optimizer =  optim.Adam(rudder_net.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    rudder_net = nn.DataParallel(rudder_net, device_ids=range(torch.cuda.device_count())).cuda()

    criteria = nn.L1Loss()
    for step, data in enumerate(dataloader):
        obs, action, done_mask, reward = [d.cuda() for d in data]
        optimizer.zero_grad()
        rewards_hat = rudder_net(obs, action) # [batch_size, seq_len, 1]
        reward_hat = torch.sum(rewards_hat[:, :, 0] * done_mask, 1, keepdim=True)
        loss = criteria(reward_hat, reward)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rudder_net.module.parameters(), 10)
        optimizer.step()

        print('[{:5d}] loss: {:.4f}\n'.format(step + 1, loss.item()))

        if step % 2000 == 1999:
            torch.save({
                'step': step + 1,
                'model_state_dict': rudder_net.module.state_dict()
            }, save_dir)
            print('model saved')

if __name__ == '__main__':
    train_rudder_mario()
