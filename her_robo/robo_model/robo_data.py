import pickle
import tensorflow as tf
from glob import glob
import os
import numpy as np
from dynamics.config import config
from multiprocessing import Process

block_size = config["block_size"]
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

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def Rollout_To_TF(rollout_path='/shared/hxu/project/prior_rl_data/', savedir = '/shared/hxu/project/prior_rl_data/', n_worker = 15):
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
        with open(path, 'rb') as f:
            episodes = pickle.load(f)
            # for episode in episodes:
            #     episode['obs'] = episode['obs'][1:, :, :, -1]

        tfrecords_filename = os.path.join(savedir, str(id)+'.tfrecord')
        writer = tf.io.TFRecordWriter(tfrecords_filename)
        state_stats = []


        for progress, episode in enumerate(episodes):
            print('[{}] {} / {}'.format(id, progress, len(episodes)))
            epi_len = len(episode['state'])
            eps_state = episode['state']
            eps_state_next = episode['state_next']
            eps_action = episode['action']
            state_stats.append(eps_state)

            features = {'state': float_feature(eps_state),
                        'action_input': float_feature(eps_action),
                        'state_next': float_feature(eps_state_next),
                        }

            data = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(data.SerializeToString())

        stats = np.array(np.mean(np.stack(state_stats), axis=0), np.std(np.stack(state_stats), axis=0))
        np.save(os.path.join(savedir, f'stats.npy'), stats)
        writer.close()


state_dim = 1
action_dim = 1

def dynamics_dataset(data_dir='{}/Datasets/MarioDynamics'.format(os.getenv('DATASET_ROOT'))):
    filenames = glob(os.path.join(data_dir, '*.tfrecord'))
    print('='*80)
    print(filenames)

    feature_description = {
        'state_input': tf.io.FixedLenFeature([state_dim], tf.float32),
        'action_input': tf.io.FixedLenFeature([action_dim], tf.float32),
        'state_prime': tf.io.FixedLenFeature([state_dim], tf.float32),
    }

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        record = tf.io.parse_single_example(example_proto, feature_description)
        return record['state_input'], record['action_input'], record['gt']

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function, num_parallel_calls=40)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=80000)
    dataset = dataset.batch(64)
    dataset = dataset.prefetch(1)
    dataset = dataset.make_one_shot_iterator()
    return dataset

if __name__ == "__main__":
    Rollout_To_TF(rollout_path='{}/Datasets/Mario'.format(os.getenv('DATASET_ROOT')), savedir='{}/Datasets/MarioDynamicsNoPos'.format(os.getenv('DATASET_ROOT')), n_worker=30)