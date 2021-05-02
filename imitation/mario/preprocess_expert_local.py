import pickle
import tensorflow as tf
from glob import glob
import os
import numpy as np
from config.mario_config import config
from multiprocessing import Process

block_size = config["block_size"]
mario_size = config["mario_size"]
screen_H = config["screen_H"]
screen_W = config["screen_W"]
seq_len = config["seq_len"]
down_sample = config["down_sample"]


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def Rollout_To_TF(rollout_path='{}/Datasets/MarioRawExpert'.format(os.getenv('DATASET_ROOT')), savedir = '{}/Datasets/MarioImitationExpert'.format(os.getenv('DATASET_ROOT')), n_worker = 10):
    episode_paths = glob(os.path.join(rollout_path, '*.pkl'))
    from random import shuffle
    shuffle(episode_paths)
    n_worker = min(n_worker, len(episode_paths))
    episode_per_worker = len(episode_paths) // n_worker

    processes = []
    for i in range(n_worker):
        start, end = i * episode_per_worker, (i + 1) * episode_per_worker
        if i==n_worker - 1:
            end = len(episode_paths)
        processes.append(Process(target=Conversion_Thread, args=(episode_paths[start: end], savedir, range(start, end))))

    for p in processes:
        p.start()


def Conversion_Thread(episode_paths, savedir, ids):
    scale_x = screen_W / 256
    scale_y = screen_H / 240

    for path, id in zip(episode_paths, ids):
        with open(path, 'rb') as f:
            episodes = pickle.load(f)
            for episode in episodes:
                episode['obs'] = episode['obs'][1:, :, :, -1]

        tfrecords_filename = os.path.join(savedir, str(id)+'.tfrecord')
        writer = tf.io.TFRecordWriter(tfrecords_filename)

        for progress, episode in enumerate(episodes):
            print('[{}] {} / {}'.format(id, progress, len(episodes)))
            obs = episode['obs'][:-1]
            info = episode['info']

            if info[-1]['x_pos'] - info[0]['x_pos'] > 2000:
                action = np.array([i['taken_action'] for i in info][1:])
                trajectory = np.array([(i['screen_x_pos'] * scale_x, (274 - i['y_pos']) * scale_y) for i in info])[:-1]

                trunc_death = 0 if np.random.random() < 0.5 else np.random.randint(1, 3)
                obs, action, trajectory, pad = sample_frames(obs, action, trajectory, seq_len, trunc_death)

                _, local_obses = get_block(obs, trajectory)
                action = action.astype(np.int64)
                for o, a, lo in zip(obs, action, local_obses):
                    features = {'obs': float_feature(o.flatten()),
                                'action': int64_feature([a]),
                                'local_obs': float_feature(lo.flatten())
                                }

                    data = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(data.SerializeToString())


        writer.close()


def sample_frames(obs, action, trajectory, seq_len, trunc_death):
    length = len(action)

    if not trunc_death:
        idxes = list(range(0, length, down_sample)) + [length - 1]
    else:
        idxes = list(range(0, length - trunc_death, down_sample))
    sample_len = len(idxes)
    return obs[idxes], action[idxes], trajectory[idxes], seq_len - sample_len

def get_block(obs, trajectory):
    blocks = []
    local_obs = []
    for o, t in zip(obs, trajectory):
        x = int(t[0] + 3)
        y = int(t[1] + 1)
        fov = 2 * block_size + mario_size
        left = x - fov // 2
        offset_left = max(0 - left, 0)
        right = left + fov
        offset_right = min(screen_W - right, 0)
        top = y - fov // 2
        offset_top = max(0 - top, 0)
        bottom = top + fov
        offset_bottom = min(screen_H - bottom, 0)
        canvas = np.zeros((fov, fov))
        # print(top, offset_top, bottom, offset_bottom, left, offset_left, right, offset_right)
        # print('=====================================')
        canvas[offset_top: max(fov + offset_bottom, 0), offset_left: max(fov + offset_right, 0)] = \
            o[top + offset_top: bottom + offset_bottom, left + offset_left:right + offset_right]
        local_obs.append(canvas)

    return np.array(blocks) / 255.0 * 2 - 1, np.array(local_obs) / 255.0 * 2 - 1

if __name__ == "__main__":
    Rollout_To_TF(rollout_path='{}/Datasets/MarioRawExpert'.format(os.getenv('DATASET_ROOT')), savedir='{}/Datasets/MarioImitationExpertLocal'.format(os.getenv('DATASET_ROOT')), n_worker=15)
