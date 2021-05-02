import pickle
import tensorflow as tf
from glob import glob
import os
import numpy as np
from multiprocessing import Process

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def Rollout_To_TF(rollout_path='{}/Datasets/MarioRawExpert'.format(os.getenv('DATASET_ROOT')), savedir = '{}/Datasets/MarioImitationExpert'.format(os.getenv('DATASET_ROOT')), n_worker = 10):
    episode_paths = glob(os.path.join(rollout_path, '*.pkl'))
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


    for path, id in zip(episode_paths, ids):
        stat = []
        with open(path, 'rb') as f:
            episodes = pickle.load(f)

        tfrecords_filename = os.path.join(savedir, str(id)+'.tfrecord')
        writer = tf.io.TFRecordWriter(tfrecords_filename)

        for progress, episode in enumerate(episodes):
            print('[{}] {} / {}'.format(id, progress, len(episodes)))
            info = episode['info']
            if info[-1]['x_pos'] - info[0]['x_pos'] > 2400:
                stat.append(info[-1]['x_pos'] - info[0]['x_pos'])
                obs = episode['obs'][:-1]
                action = np.array([i['taken_action'] for i in info])

                assert len(obs) == len(action)

                obs = obs / 255.0 * 2 - 1

                for o, a in zip(obs, action):
                    features = {'obs': float_feature(o.flatten()),
                                'action': int64_feature([a])
                               }

                    data = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(data.SerializeToString())

        np.save(os.path.join(savedir, str(id)+'.npy'), np.array(stat))
        writer.close()


if __name__ == "__main__":
    Rollout_To_TF(rollout_path='{}/Datasets/MarioRawExpert'.format(os.getenv('DATASET_ROOT')), savedir='{}/Datasets/MarioImitationExpert'.format(os.getenv('DATASET_ROOT')), n_worker=10)