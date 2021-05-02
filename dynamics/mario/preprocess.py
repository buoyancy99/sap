import pickle
import tensorflow as tf
from glob import glob
import os
import numpy as np
from config.mario_config import config
from multiprocessing import Process

mario_size = config["mario_size"]
screen_H = config["screen_H"]
screen_W = config["screen_W"]
seq_len = config["seq_len"]
down_sample = config["down_sample"]
skip = config["skip"]
plan_step = config["plan_step"]
known_len = config["known_len"]
v_mean = config["v_mean"]
v_std = config["v_std"]
pos_mean = config["pos_mean"]
pos_std = config["pos_std"]
action_space = len(config["movement"])

def normalize_v(v, batched = True):
    if batched:
        return (v - v_mean[None, :]) / v_std[None, :]
    else:
        return (v - v_mean) / v_std

def normalize_pos(pos, batched = True):
    if batched:
        return (pos - pos_mean[None, :]) / pos_std[None, :]
    else:
        return (pos - pos_mean) / pos_std

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def Rollout_To_TF(rollout_path='{}/Datasets/MarioRaw'.format(os.getenv('DATASET_ROOT')), savedir = '{}/Datasets/MarioDynamics{}'.format(os.getenv('DATASET_ROOT'), plan_step), n_worker = 15):
    episode_paths = glob(os.path.join(rollout_path, '*.pkl'))
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
                episode['obs'] = episode['obs'][1:-1, :, :, -1]

        tfrecords_filename = os.path.join(savedir, str(id)+'.tfrecord')
        writer = tf.io.TFRecordWriter(tfrecords_filename)
        stats = []

        for progress, episode in enumerate(episodes):
            print('[{}] {} / {}'.format(id, progress, len(episodes)))
            epi_len = len(episode['obs'])
            eps_obs = episode['obs']
            eps_info = episode['info']
            eps_done = episode['done']
            eps_x = np.array([info['x_pos'] * scale_x for info in eps_info])
            eps_screen_x = np.array([info['screen_x_pos'] * scale_x for info in eps_info])
            eps_y = np.array([(274 - info['y_pos']) * scale_y for info in eps_info])
            eps_action = np.array([info['taken_action'] for info in eps_info])

            for i in range(known_len, epi_len):
                obs = eps_obs[i]
                for j in range(min(plan_step, epi_len - i)):
                    action = eps_action[i + j - 2: i + j + 2]
                    done = eps_done[i + j + 1]
                    x = eps_x[i + j]
                    y = eps_y[i + j]
                    new_x = eps_x[i + j + 1]
                    new_y = eps_y[i + j + 1]
                    dx = new_x - x
                    dy = new_y - y
                    screen_x = eps_screen_x[i + j]
                    block_input = get_block(obs, (screen_x, y))
                    gt = np.array((dx, dy))
                    vx_input = eps_x[i + j - known_len + 2: i + j + 1] - eps_x[i + j - known_len + 1: i + j]
                    vy_input = eps_y[i + j - known_len + 2: i + j + 1] - eps_y[i + j - known_len + 1: i + j]
                    v_input = normalize_v(np.array(list(zip(vx_input, vy_input))), True)
                    # posx_input = eps_screen_x[i + j - known_len + 2: i + j + 1]
                    # posy_input = eps_y[i + j - known_len + 2: i + j + 1]
                    # pos_input = normalize_pos(np.array(list(zip(posx_input, posy_input))), True)
                    pos_input = v_input.flatten()

                    stats.append([gt[0], gt[1], screen_x, y])

                    augment = 2 if done else 1
                    for _ in range(augment):
                        features = {'block_input': float_feature(block_input.flatten()),
                                    'action_input': int64_feature(action),
                                    'pos_input': float_feature(pos_input),
                                    'step': int64_feature([j]),
                                    'gt': float_feature(normalize_v(gt, False)),
                                    'done': float_feature([done])
                                    }

                        data = tf.train.Example(features=tf.train.Features(feature=features))
                        writer.write(data.SerializeToString())

        stats = np.array(stats)
        np.save(os.path.join(savedir, 'stat{}.npy'.format(id)), stats)

        writer.close()


def get_block(obs, pos):
    x = pos[0] + 3
    y = pos[1] + 1
    left = int(x - mario_size * 5 // 2)
    offset_left = max(0 - left, 0)
    right = left + 5 * mario_size
    offset_right = min(screen_W - right, 0)
    top = int(y - mario_size * 5 // 2)
    offset_top = max(0 - top, 0)
    bottom = top + 5 * mario_size
    offset_bottom = min(screen_H - bottom, 0)
    canvas = np.zeros((5 * mario_size, 5 * mario_size))
    # print(top, offset_top, bottom, offset_bottom, left, offset_left, right, offset_right)
    # print('=====================================')
    canvas[offset_top: max(5 * mario_size + offset_bottom, 0), offset_left: max(5 * mario_size + offset_right, 0)] = \
        obs[top + offset_top: bottom + offset_bottom, left + offset_left:right + offset_right]

    return canvas / 255.0 * 2 - 1

if __name__ == "__main__":
    Rollout_To_TF(rollout_path='{}/Datasets/MarioRaw'.format(os.getenv('DATASET_ROOT')), savedir='{}/Datasets/MarioDynamics{}'.format(os.getenv('DATASET_ROOT'), plan_step), n_worker=2)
