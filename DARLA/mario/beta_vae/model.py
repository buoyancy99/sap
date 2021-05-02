import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # take in 64 by 64
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1),
            nn.ReLU(), # 32
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.ReLU(), # 16
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(), # 8
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(), # 4
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(), # 2
            nn.Flatten()
        )

        self.mu = nn.Linear(512, 128)

        self.log_var = nn.Linear(512, 128)

        self.decoder_fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * 2 - 1
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = mu + torch.mul(torch.exp(log_var / 2), torch.randn_like(log_var))
        x_hat = self.decoder(self.decoder_fc(z).view(-1, 128, 2, 2))

        return x_hat, mu, log_var

    def encode(self, x):
        x = x * 2 - 1
        x = self.encoder(x)
        mu = self.mu(x)
        return mu

    def decode(self, z):
        return self.decoder(self.decoder_fc(z).view(-1, 128, 2, 2))
