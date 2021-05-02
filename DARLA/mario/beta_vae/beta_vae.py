import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
from torchvision.utils import make_grid

from DARLA.mario.dataset import DARLA_Dataset
from DARLA.mario.beta_vae.model import Model

def KL(mu, log_var):
    kl = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 1), 0)
    return kl

class BetaVAE():
    def __init__(self, dae, beta, ckpt_path='DARLA/mario/beta_vae/ckpts/latest.model'):
        self.dae = dae
        self.beta = beta
        self.vae = Model().cuda()
        if os.path.exists(ckpt_path):
            print('loading trained model')
            self.vae.load_state_dict(torch.load(ckpt_path)['model_state_dict'])


    def encode(self, x):
        x = nn.functional.interpolate(x, size=(64, 64), mode='bilinear')
        return self.vae.encode(x)

    def decode(self, z):
        return self.vae.decode(z)

    def train(self, num_epochs=3, batch_size=64, lr=1e-4, save_iter = 5000):

        print('Creating dataset')
        dataloader = DataLoader(DARLA_Dataset(), batch_size=batch_size, num_workers=16, shuffle=True)
        writer = SummaryWriter('logs/DARLA/mario/BetaVAE')
        print('Training ß-VAE...')

        optimizer = optim.Adam(self.vae.parameters(), lr=lr)
        criteria = nn.MSELoss()

        step = 0
        for epoch in range(num_epochs):
            for data in dataloader:
                step += 1
                img, img_corrupt = [d.cuda() for d in data]
                img = nn.functional.interpolate(img, size=(64, 64), mode='bilinear')
                img_hat, mu, log_var = self.vae(img)
                pixel_loss = criteria(self.dae.encode(img_hat), self.dae.encode(img))
                KL_loss = self.beta * KL(mu, log_var)
                loss = pixel_loss + KL_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 20 == 0:
                    text = '[{:5d}] loss: {:.3f} pixel_loss: {:.3f} KL_loss: {:.3f}'.format(step, loss.item(), pixel_loss.item(), KL_loss.item())
                    print(text)
                    writer.add_scalar('loss', loss.item(), step)
                    writer.add_scalar('pixel_loss', pixel_loss.item(), step)
                    writer.add_scalar('KL_loss', KL_loss.item(), step)

                if step % 1000 == 0:
                    sample_img_batch = img_hat.detach().cpu()[:16]
                    sample_img_batch = make_grid(sample_img_batch, 4)
                    sample_img_batch = nn.functional.interpolate(sample_img_batch[None], size=(1024, 1024))[0]
                    writer.add_image('VAE_sample', sample_img_batch, step)

                    sample_img_batch = self.dae.dae(img_hat).detach().cpu()[:16]
                    sample_img_batch = make_grid(sample_img_batch, 4)
                    sample_img_batch = nn.functional.interpolate(sample_img_batch[None], size=(1024, 1024))[0]
                    writer.add_image('VAE_DAE_sample', sample_img_batch, step)

                if step % save_iter == 0:
                    name = 'DARLA/mario/beta_vae/ckpts/latest.model'
                    torch.save({
                        'model_state_dict': self.vae.state_dict(),
                        'step': step,
                    }, name)
                    print('beta-VAE model saved')
