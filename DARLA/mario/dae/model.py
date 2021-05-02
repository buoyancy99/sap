import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1),
            nn.ReLU(),  # 32
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.ReLU(),  # 16
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),  # 8
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),  # 4
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),  # 2
            nn.Flatten(),
            nn.Linear(512, 128)
        )

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
        x = x + torch.randn_like(x)
        z = self.encoder(x)
        return self.decoder(self.decoder_fc(z).view(-1, 128, 2, 2))

    def encode(self, x):
        x = x * 2 - 1
        x = x + torch.randn_like(x)
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(self.decoder_fc(z).view(-1, 128, 2, 2))
