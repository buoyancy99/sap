import torch
from torchvision import transforms
import pickle
import os
from glob import glob

class DARLA_Dataset(torch.utils.data.Dataset):
    def __init__(self, rollout_path='{}/Datasets/MarioRaw'.format(os.getenv('DATASET_ROOT'))):
        episode_paths = glob(os.path.join(rollout_path, '*.pkl'))

        self.data = []
        cutoff = 1000000
        for path in episode_paths:
            print('loading', path)
            with open(path, 'rb') as f:
                episodes = pickle.load(f)
                for episode in episodes:
                    self.data.extend(episode['obs'][:, :, :, 1:])
            if len(self.data) >= cutoff:
                break

        self.totensor = transforms.ToTensor()
        self.random_erase = transforms.RandomErasing(p=0.9, scale=(0, 1), ratio=(0, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = self.totensor(img)
        erased = self.random_erase(img)

        return img, erased

if __name__=='__main__':
    d = DARLA_Dataset()
    t = d.__getitem__(5)
    print(t.shape)