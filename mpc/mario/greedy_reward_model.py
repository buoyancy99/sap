from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import numpy as np
from config.mario_config import config

seq_len = config["plan_step"]
down_sample = config["down_sample"]
block_size = config["block_size"]
num_envs = config["num_envs"]
action_space = len(config["movement"])
follow_steps = config["follow_steps"]
gamma = config["gamma"]
trained_on = config["trained_on"]

def make_model(checkpoint_path):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    conv1_w = tf.convert_to_tensor(reader.get_tensor('conv1/w'))
    conv1_b = tf.convert_to_tensor(reader.get_tensor('conv1/b'))
    conv2_w = tf.convert_to_tensor(reader.get_tensor('conv2/w'))
    conv2_b = tf.convert_to_tensor(reader.get_tensor('conv2/b'))
    fc1_w = tf.convert_to_tensor(reader.get_tensor('fc1/w'))
    fc1_b = tf.convert_to_tensor(reader.get_tensor('fc1/b'))
    fc2_w = tf.convert_to_tensor(reader.get_tensor('fc2/w'))
    fc2_b = tf.convert_to_tensor(reader.get_tensor('fc2/b'))

    blocks_input = tf.placeholder(tf.float32, [None, 8, block_size, block_size, 1])
    batch_size = tf.cast(tf.shape(blocks_input)[0], tf.int64)
    blocks = tf.reshape(blocks_input, (batch_size * 8, block_size, block_size, 1))

    pos_idx = tf.tile(tf.range(8, dtype=tf.int64), (batch_size, ))
    pos_idx = tf.reshape(pos_idx, (batch_size  * 8, 1))

    idx = tf.range(batch_size * 8, dtype=tf.int64)
    idx = tf.reshape(idx, (batch_size * 8, 1))

    mask_idx = tf.concat([idx, pos_idx], 1)
    mask_value = tf.ones((batch_size * 8,))

    mask = tf.sparse.SparseTensor(mask_idx, mask_value, tf.convert_to_tensor((batch_size * 8, 8), dtype=tf.int64))

    X = tf.nn.relu(tf.nn.conv2d(blocks, filter=conv1_w, strides=[1, 2, 2, 1], padding="SAME") + conv1_b)
    X = tf.nn.relu(tf.nn.conv2d(X, filter=conv2_w, strides=[1, 2, 2, 1], padding="SAME") + conv2_b)
    X = tf.reshape(X, (-1, 16 * 3 * 3))
    X = tf.nn.relu(tf.matmul(X, fc1_w) + fc1_b)
    X = tf.matmul(X, fc2_w) + fc2_b

    X = tf.reshape(X, (batch_size * 8, 8, action_space))
    X = X * tf.reshape(tf.sparse.to_dense(mask), (batch_size * 8, 8, 1))
    X = tf.reshape(X, (batch_size, 8, 8, action_space))

    output = tf.reduce_sum(X, [1, 2])

    return blocks_input, output

class greedy_reward_predictor:
    def __init__(self, checkpoint_path = './prior_learning/mario/ckpts/reward_{}'.format(trained_on), reward_type = False):
        self.blocks_input, self.output = make_model(checkpoint_path)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

    def predict(self, infos):
        # info is a list of list
        blocks = infos[0]['blocks'] * 2 - 1
        rewards = self.sess.run(self.output, feed_dict={self.blocks_input: blocks[None]})[0]
        dist = np.exp(rewards - np.max(rewards))
        dist = dist / np.sum(dist)

        best_action = np.random.choice(5, 1, p=dist)[0]
        return best_action


    def close(self):
        self.sess.close()


