from glob import glob
import os
from config.mario_config import config
import tensorflow as tf

block_size = config["block_size"]
screen_H = config["screen_H"]
screen_W = config["screen_W"]
seq_len = config["seq_len"]
down_sample = config["down_sample"]

def imitation_dataset(data_dir='{}/Datasets/MarioImitationDarla'.format(os.getenv('DATASET_ROOT'))):
    filenames = glob(os.path.join(data_dir, '*.tfrecord'))
    print('='*80)
    print(filenames)

    feature_description = {
        'obs': tf.io.FixedLenFeature([128], tf.float32),
        'action': tf.io.FixedLenFeature([], tf.int64)
    }

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        record = tf.io.parse_single_example(example_proto, feature_description)
        return record['obs'], record['action']

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function, num_parallel_calls=64)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=80000)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(2)
    dataset = dataset.make_one_shot_iterator()

    return dataset
