from glob import glob
import os
from config.mario_config import config
import tensorflow as tf

mario_size = config["mario_size"]
screen_H = config["screen_H"]
screen_W = config["screen_W"]
seq_len = config["seq_len"]
down_sample = config["down_sample"]
plan_step = config["plan_step"]
known_len = config["known_len"]
action_space = len(config["movement"])

def dynamics_dataset(data_dir='{}/Datasets/MarioDynamics{}'.format(os.getenv('DATASET_ROOT'), plan_step)):
    filenames = glob(os.path.join(data_dir, '*.tfrecord'))
    print('='*80)
    print(filenames)

    feature_description = {
        'block_input': tf.io.FixedLenFeature([5 * mario_size, 5 * mario_size, 1], tf.float32),
        'action_input': tf.io.FixedLenFeature([4], tf.int64),
        'pos_input': tf.io.FixedLenFeature([2 * (known_len - 1)], tf.float32),
        'step': tf.io.FixedLenFeature([], tf.int64),
        'gt': tf.io.FixedLenFeature([2], tf.float32),
        'done': tf.io.FixedLenFeature([1], tf.float32)
    }

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        record = tf.io.parse_single_example(example_proto, feature_description)
        return record['block_input'], record['action_input'], record['pos_input'], record['step'], record['gt'], record['done']

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function, num_parallel_calls=64)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=100000)
    dataset = dataset.batch(64)
    dataset = dataset.prefetch(2)
    dataset = dataset.make_one_shot_iterator()

    return dataset

