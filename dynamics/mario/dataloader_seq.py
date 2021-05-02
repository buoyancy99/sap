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

def dynamics_seq_dataset(data_dir='{}/Datasets/MarioDynamics{}'.format(os.getenv('DATASET_ROOT'), plan_step)):
    filenames = glob(os.path.join(data_dir, '*.tfrecord'))
    print('='*80)
    print(filenames)

    sequence_feature_description = {
        'block_input': tf.io.FixedLenSequenceFeature([5 * mario_size, 5 * mario_size, 1], tf.float32),
        'action_input': tf.io.FixedLenSequenceFeature([4], tf.int64),
        'pos_input': tf.io.FixedLenSequenceFeature([2 * (known_len - 1)], tf.float32),
        'step': tf.io.FixedLenSequenceFeature([], tf.int64),
        'gt': tf.io.FixedLenSequenceFeature([2], tf.float32),
        'done': tf.io.FixedLenSequenceFeature([1], tf.float32)
    }

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        context_record, sequence_record = tf.io.parse_single_sequence_example(example_proto, sequence_features=sequence_feature_description)
        return sequence_record['block_input'], sequence_record['action_input'], sequence_record['pos_input'], sequence_record['step'], sequence_record['gt'], sequence_record['done'], tf.ones(tf.shape(sequence_record['step']))

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function, num_parallel_calls=64)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=100000)
    dataset = dataset.padded_batch(64, (
        [None, 5 * mario_size, 5 * mario_size, 1],
        [None, 4],
        [None, 2 * (known_len - 1)],
        [None],
        [None, 2],
        [None, 1],
        [None]))
    dataset = dataset.prefetch(2)
    dataset = dataset.make_one_shot_iterator()

    return dataset

