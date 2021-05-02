from glob import glob
import os
from config.mario_config import config
import tensorflow as tf

block_size = config["block_size"]
screen_H = config["screen_H"]
screen_W = config["screen_W"]
seq_len = config["seq_len"]
down_sample = config["down_sample"]

def mario_dataset(data_dir='{}/Datasets/MarioDemo'.format(os.getenv('DATASET_ROOT'))):
    filenames = glob(os.path.join(data_dir, '*.tfrecord'))
    print('='*80)
    print(filenames)

    feature_description = {
        'blocks': tf.io.FixedLenFeature([seq_len, 8, block_size, block_size], tf.float32),
        'action': tf.io.FixedLenFeature([seq_len], tf.int64),
        'mask': tf.io.FixedLenFeature([seq_len], tf.float32),
        'total_reward': tf.io.FixedLenFeature([], tf.float32)
    }

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        record = tf.io.parse_single_example(example_proto, feature_description)
        return record['blocks'], record['action'], record['mask'], record['total_reward']


    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function, num_parallel_calls=128)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=20)
    dataset = dataset.batch(256)
    dataset = dataset.prefetch(1)
    dataset = dataset.make_initializable_iterator()

    return dataset

