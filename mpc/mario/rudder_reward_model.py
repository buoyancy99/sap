import numpy as np
import torch
from config.mario_config import config
from prior_learning.mario.rudder_model import Rudder_Net

seq_len = config["seq_len"]
block_size = config["block_size"]
num_envs = config["num_envs"]
action_space = len(config["movement"])
follow_steps = config["follow_steps"]
gamma = config["gamma"]
trained_on = config["trained_on"]

class reward_predictor:
    def __init__(self, checkpoint_path = './prior_learning/mario/ckpts/rudder_reward_{}.model'.format(trained_on)):
        self.net = Rudder_Net()
        state_dict = torch.load(checkpoint_path)
        self.net.load_state_dict(state_dict['model_state_dict'])
        self.net.eval()
        self.net = self.net.cuda()

        self.reset()

    def reset(self):
        self.obs_buffer = np.zeros((num_envs, seq_len, 1, 30, 30), dtype=np.float32)
        self.action_buffer = np.zeros((num_envs, seq_len), dtype=np.int64)
        self.mask_buffer = np.zeros((num_envs, seq_len), dtype=np.float32)
        self.not_done = np.ones(num_envs)
        self.buffer_top = 0
        self.plan_buffer_top = 0

    def predict(self):
        # info is a list of list
        self.obs_buffer[:, self.plan_buffer_top - 1] = 0
        observation = torch.from_numpy(self.obs_buffer * 2 - 1).cuda()
        action = torch.from_numpy(np.eye(5, dtype=np.float32)[self.action_buffer]).cuda()
        rewards_hat = self.net(observation, action).detach().cpu().numpy()[:, :, 0] * self.mask_buffer
        best_id = np.argmax(rewards_hat[:, self.plan_buffer_top - 2])
        best_action = self.action_buffer[best_id, :follow_steps]
        self.buffer_top += follow_steps
        self.plan_buffer_top = self.buffer_top
        self.obs_buffer[:, self.buffer_top:] = 0
        self.action_buffer[:, self.buffer_top:] = 0
        self.not_done = np.ones(num_envs)

        return best_action

    def update(self, infos, dones = np.zeros(num_envs)):
        batch_obses = []
        batch_actions = []

        if 'action_taken' in infos[0].keys():
            for env_id, info in enumerate(infos):
                if self.not_done[env_id]:
                    batch_actions.append(info['action_taken'])
                else:
                    batch_actions.append(0)
            self.action_buffer[:, self.plan_buffer_top - 1] = batch_actions
        else:
            assert self.plan_buffer_top == 0

        self.not_done = self.not_done * (1 - dones)

        for env_id, info in enumerate(infos):
            if self.not_done[env_id]:
                batch_obses.append(info['local_obs'])
            else:
                batch_obses.append(np.zeros((1, 30, 30)))

        self.obs_buffer[:, self.plan_buffer_top] = np.stack(batch_obses, 0)
        self.mask_buffer[:, self.plan_buffer_top] = self.not_done

        self.plan_buffer_top += 1

    def close(self):
        pass


