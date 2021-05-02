import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
from torchvision.utils import make_grid

from DARLA.mario.dae.model import Model
from DARLA.mario.dataset import DARLA_Dataset

class DAE():
    def __init__(self, ckpt_path='DARLA/mario/dae/ckpts/latest.model'):
        self.dae = Model().cuda()
        if ckpt_path and os.path.exists(ckpt_path):
            print('loading trained model')
            self.dae.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

    def encode(self, x):
        x = nn.functional.interpolate(x, size=(64, 64), mode='bilinear')
        return self.dae.encode(x)

    def decode(self, z):
        return self.dae.decode(z)

    def train(self, num_epochs=3, batch_size=64, lr=1e-3, save_iter = 5000):
        print('Creating dataset')
        dataloader = DataLoader(DARLA_Dataset(), batch_size=batch_size, num_workers=16, shuffle=True)
        writer = SummaryWriter('logs/DARLA/mario/DAE')
        print('Training DAE...')

        optimizer = optim.Adam(self.dae.parameters(), lr=lr)
        criteria = nn.MSELoss()

        step = 0
        for epoch in range(num_epochs):
            for data in dataloader:
                step += 1
                img, img_corrupt = [d.cuda() for d in data]
                img = nn.functional.interpolate(img, size=(64, 64), mode='bilinear')
                img_corrupt = nn.functional.interpolate(img_corrupt, size=(64, 64), mode='bilinear')
                img_hat = self.dae(img_corrupt)
                loss = criteria(img_hat, img) * 200
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 20 == 0:
                    text = '[{:5d}] loss: {:.3f}'.format(step, loss.item())
                    print(text)
                    writer.add_scalar('loss', loss.item(), step)

                if step % 1000 == 0:
                    sample_img_batch = img_hat.detach().cpu()[:16]
                    sample_img_batch = make_grid(sample_img_batch, 4)
                    sample_img_batch = nn.functional.interpolate(sample_img_batch[None], size=(1024, 1024))[0]
                    writer.add_image('DAE_sample', sample_img_batch, step)

                if step % save_iter == 0:
                    name = 'DARLA/mario/dae/ckpts/latest.model'
                    torch.save({
                        'model_state_dict': self.dae.state_dict(),
                        'step': step,
                    }, name)
                    print('DAE model saved')


