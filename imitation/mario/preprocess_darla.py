from glob import glob
import os
import tensorflow as tf
from tfrecord_lite import decode_example
import torch
import torch.nn as nn
import numpy as np

from DARLA.mario.beta_vae.model import Model as Beta_VAE_Net

BATCH_SIZE = 16384

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_data(data_dir='{}/Datasets/MarioImitation'.format(os.getenv('DATASET_ROOT')), save_dir='{}/Datasets/MarioImitationDarla'.format(os.getenv('DATASET_ROOT')), ckpt_path='DARLA/mario/beta_vae/ckpts/latest.model'):
    data_paths = glob(os.path.join(data_dir, '*.tfrecord'))

    vae = Beta_VAE_Net().cuda()
    print('loading trained model')
    vae.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    for param in vae.parameters():
        param.requires_grad = False
    print('model loaded')

    for i, data_path in enumerate(data_paths):
        print('{} / {} loading {}'.format(i + 1, len(data_paths), data_path))
        dataloader = tf.python_io.tf_record_iterator(data_path)
        action_batch = []
        obs_batch = []
        frecords_filename = os.path.join(save_dir, str(i) + '.tfrecord')
        writer = tf.io.TFRecordWriter(frecords_filename)

        for progress, d in enumerate(dataloader):
            print('{} , {}'.format(i, progress))
            d = decode_example(d)
            obs_batch.append(d['obs'].reshape((84, 84, 4))[:, :, 1:])
            action_batch.append(d['action'])

            if len(obs_batch) == BATCH_SIZE:
                obs = (np.stack(obs_batch, 0) + 1) / 2
                obs = torch.from_numpy(obs).float().cuda()
                obs = obs.permute(0, 3, 1, 2)
                obs = nn.functional.interpolate(obs, size=(64, 64), mode='bilinear')
                encoding = vae.encode(obs).detach().cpu().numpy()

                for e, a in zip(encoding, action_batch):
                    features = {'obs': float_feature(e.flatten()),
                                'action': int64_feature([a])
                                }

                    writer.write(tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString())

                obs_batch = []
                action_batch = []

if __name__=='__main__':
    convert_data()

