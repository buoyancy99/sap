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

def Rollout_To_TF(rollout_path='{}/Datasets/Mario'.format(os.getenv('DATASET_ROOT')), savedir = '{}/Datasets/MarioDemo'.format(os.getenv('DATASET_ROOT')), n_worker = 10):
    episode_paths = glob(os.path.join(rollout_path, '*.pkl'))
    from random import shuffle
    shuffle(episode_paths)
    n_worker = min(n_worker, len(episode_paths))
    episode_per_worker = len(episode_paths) // n_worker
    stats = get_stats(episode_paths)
    np.save(os.path.join(savedir, 'stat.npy'), stats)
    stats  = np.load(os.path.join(savedir, 'stat.npy'))
    print('collected from {} stats, max {}'.format(len(stats), np.max(stats)))

    processes = []
    for i in range(n_worker):
        start, end = i * episode_per_worker, (i + 1) * episode_per_worker
        if i==n_worker - 1:
            end = len(episode_paths)
        processes.append(Process(target=Conversion_Thread, args=(episode_paths[start: end], savedir, range(start, end), np.copy(stats))))

    for p in processes:
        p.start()

def Conversion_Thread(episode_paths, savedir, ids, stats):
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
            action = np.array([i['taken_action'] for i in info][1:])
            trajectory = np.array([(i['screen_x_pos'] * scale_x, (274 - i['y_pos']) * scale_y) for i in info])[:-1]

            trunc_death = 0 if np.random.random() < 0.5 else np.random.randint(1, 3)
            obs, action, trajectory, pad = sample_frames(obs, action, trajectory, seq_len, trunc_death)

            if pad < 0:
                continue

            if trunc_death:
                total_reward = info[-1-trunc_death]['x_pos'] - info[0]['x_pos']
                stats_mask = stats > total_reward
                num_longer_eps = np.sum(stats_mask)
                mean_future_reward = np.sum(stats * stats_mask / num_longer_eps)  - total_reward if num_longer_eps > 0 else 0
                total_reward += mean_future_reward
            else:
                total_reward = (info[-1]['x_pos'] - info[0]['x_pos'])

            total_reward *= scale_x

            blocks, local_obses = get_block(obs, trajectory, pad)
            local_obses_no_mario = local_obses.copy()
            local_obses_no_mario[:, block_size: -block_size, block_size: -block_size] = 0
            padded_action = np.zeros(seq_len)
            padded_action[:len(action)] = action
            done_mask = np.zeros(seq_len)
            done_mask[len(action) - 1] = 1
            action = padded_action
            mask = np.zeros(seq_len)
            mask[:len(action)] = 1

            blocks = blocks.flatten().astype(np.float32)
            local_obses = local_obses.flatten().astype(np.float32)
            local_obses_no_mario = local_obses_no_mario.flatten().astype(np.float32)
            action = action.flatten().astype(np.int64)
            mask = mask.flatten().astype(np.float32)
            done_mask = done_mask.flatten().astype(np.float32)
            total_reward = total_reward.flatten().astype(np.float32)

            features = {'blocks': float_feature(blocks),
                        'action': int64_feature(action),
                        'mask': float_feature(mask),
                        'done_mask': float_feature(done_mask),
                        'total_reward': float_feature(total_reward),
                        'local_obs': float_feature(local_obses),
                        'local_obses_no_mario': float_feature(local_obses_no_mario)
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

def get_block(obs, trajectory, pad):
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
        x, y = fov // 2, fov // 2
        tl = canvas[:block_size, :block_size]
        tm = canvas[:block_size, x - block_size // 2: x + block_size // 2]
        tr = canvas[:block_size, -block_size:]
        ml = canvas[y - block_size // 2: y + block_size // 2, :block_size]
        mr = canvas[y - block_size // 2: y + block_size // 2, -block_size:]
        bl = canvas[-block_size:, :block_size]
        bm = canvas[-block_size:, x - block_size // 2: x + block_size // 2]
        br = canvas[-block_size:, -block_size:]
        blocks.append(np.stack([tl, tm , tr, ml, mr, bl, bm, br], axis=0))
        local_obs.append(canvas)

    for _ in range(pad):
        blocks.append(np.zeros((8, block_size, block_size)))
        local_obs.append(np.zeros((fov, fov)))

    return np.array(blocks) / 255.0 * 2 - 1, np.array(local_obs) / 255.0 * 2 - 1

def get_stats(episode_paths):
    print('geting stats')
    stat = []
    for path in episode_paths:
        print('collecting data from {}'.format(path))
        with open(path, 'rb') as f:
            episodes = pickle.load(f)
            for episode in episodes:
                info = episode['info']
                total_reward = info[-1]['x_pos'] - info[0]['x_pos']
                stat.append(total_reward)
    print('stat collected')

    return np.array(stat)


if __name__ == "__main__":
    Rollout_To_TF(rollout_path='{}/Datasets/MarioRaw'.format(os.getenv('DATASET_ROOT')), savedir='{}/Datasets/MarioDemo'.format(os.getenv('DATASET_ROOT')), n_worker=15)
