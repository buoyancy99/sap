from torch.utils.data import Dataset
import tensorflow as tf
from tfrecord_lite import decode_example
from glob import glob
import os
import numpy as np

class Rudder_Mario_Dataset(Dataset):
    def __init__(self, data_dir='{}/Datasets/MarioDemo'.format(os.getenv('DATASET_ROOT'))):
        super(Rudder_Mario_Dataset, self).__init__()
        self.data_paths = glob(os.path.join(data_dir, '*.tfrecord'))
        data = []
        for i, data_path in enumerate(self.data_paths):
            data += list(tf.python_io.tf_record_iterator(data_path))
            print('{} / {} loading {}'.format(i + 1, len(self.data_paths), data_path))
        self.data = data
        self.one_hot_matrix = np.eye(5, dtype=np.float32)

    def __len__(self):
        return 100000000

    def __getitem__(self, idx):
        data_dict = decode_example(self.data[idx % len(self.data)])
        local_obs = data_dict['local_obs'].reshape((-1, 1, 30, 30))
        action = self.one_hot_matrix[data_dict['action']]
        done_mask = data_dict['done_mask']
        reward = data_dict['total_reward']
        return local_obs,  action, done_mask, reward



