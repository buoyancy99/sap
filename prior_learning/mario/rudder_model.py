import torch
import torch.nn as nn
from widis_lstm_tools.nn import LSTMLayer
from config.mario_config import config

class Rudder_Net(torch.nn.Module):
    def __init__(self, n_lstm = 64):
        super().__init__()
        self.obs_image_size = 30
        self.action_space_size = 5

        # Assume input image to be self.obs_image_size by self.obs_image_size
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(1, 3, 3, 2, 1),
            nn.ReLU(), #15
            nn.Conv2d(3, 8, 3, 2, 1),
            nn.ReLU(), #8
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.ReLU(), #4
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(), #2
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_space_size, 32),
            nn.ReLU()
        )

        self.lstm1 = LSTMLayer(
            in_features= 96, out_features=n_lstm, inputformat='NLC',
            w_ci=(torch.nn.init.xavier_normal_, False),
            w_ig=(False, torch.nn.init.xavier_normal_),
            w_og=False, b_og=False,
            w_fg=False, b_fg=False,
            a_out=lambda x: x
        )

        self.fc_out = torch.nn.Linear(n_lstm, 1)

    def forward(self, observations, actions):
        seq_len = actions.shape[1]
        obs_encoding = self.obs_encoder(observations.view(-1, 1, self.obs_image_size, self.obs_image_size))
        action_encoding = self.action_encoder(actions.view(-1, self.action_space_size))

        encoding = torch.cat([obs_encoding, action_encoding], -1).view(-1, seq_len, 96)
        lstm_out, *_ = self.lstm1(encoding, return_all_seq_pos=True)
        rewards_hat = self.fc_out(lstm_out)

        return rewards_hat
