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

    blocks_input = tf.placeholder(tf.float32, [None, seq_len, 8, block_size, block_size, 1])
    actions_input = tf.placeholder(tf.int64, [None, seq_len, ])
    seq_mask = tf.placeholder(tf.float32, [None, seq_len, ])

    batch_size = tf.cast(num_envs, tf.int64)
    blocks = tf.reshape(blocks_input, (batch_size * seq_len * 8, block_size, block_size, 1))

    pos_idx = tf.tile(tf.range(8, dtype=tf.int64), (batch_size * seq_len,))
    pos_idx = tf.reshape(pos_idx, (batch_size * seq_len * 8, 1))

    actions = tf.reshape(actions_input, (batch_size * seq_len, 1))
    action_idx = tf.tile(actions, (1, 8))
    action_idx = tf.reshape(action_idx, (batch_size * seq_len * 8, 1))

    idx = tf.range(batch_size * seq_len * 8, dtype=tf.int64)
    idx = tf.reshape(idx, (batch_size * seq_len * 8, 1))

    mask_idx = tf.concat([idx, pos_idx, action_idx], 1)
    mask_value = tf.ones((batch_size * seq_len * 8,))

    mask = tf.sparse.SparseTensor(mask_idx, mask_value, tf.convert_to_tensor((batch_size * seq_len * 8, 8, action_space), dtype=tf.int64))

    X = tf.nn.relu(tf.nn.conv2d(blocks, filter=conv1_w, strides=[1, 2, 2, 1], padding="SAME") + conv1_b)
    X = tf.nn.relu(tf.nn.conv2d(X, filter=conv2_w, strides=[1, 2, 2, 1], padding="SAME") + conv2_b)
    X = tf.reshape(X, (-1, 16 * 3 * 3))
    X = tf.nn.relu(tf.matmul(X, fc1_w) + fc1_b)
    X = tf.matmul(X, fc2_w) + fc2_b

    X = tf.reshape(X, (batch_size * seq_len * 8, 8, action_space))
    X = X * tf.sparse.to_dense(mask)
    X = tf.reshape(X, (batch_size, seq_len, 8, 8, action_space))
    discount_mask = tf.reshape(tf.convert_to_tensor(np.geomspace(1, gamma, num=seq_len), dtype=tf.float32), (1, seq_len))
    output = tf.reduce_sum(X, [2, 3, 4]) * seq_mask * discount_mask
    output = tf.reduce_sum(output, [1])

    return blocks_input, actions_input, seq_mask, output

def make_mbhp_model():
    blocks_input = tf.placeholder(tf.float32, [None, seq_len, 8, block_size, block_size, 1])
    actions_input = tf.placeholder(tf.int64, [None, seq_len, ])
    seq_mask = tf.placeholder(tf.float32, [None, seq_len, ])
    discount_mask = tf.reshape(tf.convert_to_tensor(np.geomspace(1, gamma, num=seq_len), dtype=tf.float32), (1, seq_len))
    output = tf.cast(actions_input > 0, tf.float32) * seq_mask * discount_mask
    output = tf.reduce_sum(output, [1])

    return blocks_input, actions_input, seq_mask, output

class reward_predictor:
    def __init__(self, checkpoint_path = './prior_learning/mario/ckpts/reward_{}'.format(trained_on), mbhp = False):
        if mbhp:
            self.blocks_input, self.actions_input, self.seq_mask, self.output = make_mbhp_model()
        else:
            self.blocks_input, self.actions_input, self.seq_mask, self.output = make_model(checkpoint_path)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.initialize_buffer()

    def initialize_buffer(self):
        self.block_buffer = np.zeros((num_envs, seq_len, 8, block_size, block_size, 1), dtype=np.float32)
        self.action_buffer = np.zeros((num_envs, seq_len), dtype=np.int64)
        self.mask_buffer = np.zeros((num_envs, seq_len), dtype=np.float32)
        self.not_done = np.ones(num_envs)
        self.buffer_top = 0

    def predict(self):
        # info is a list of list
        rewards = self.sess.run(self.output, feed_dict={self.blocks_input: self.block_buffer, self.actions_input: self.action_buffer, self.seq_mask: self.mask_buffer})
        best_id = np.argmax(rewards)
        best_action = self.action_buffer[best_id, :follow_steps]
        self.initialize_buffer()

        return best_action


    def update(self, infos, dones = np.zeros(num_envs)):
        batch_blocks = []
        batch_actions = []

        if 'action_taken' in infos[0].keys():
            for env_id, info in enumerate(infos):
                if self.not_done[env_id]:
                    batch_actions.append(info['action_taken'])
                else:
                    batch_actions.append(0)
            self.action_buffer[:, self.buffer_top - 1] = batch_actions
        else:
            assert self.buffer_top == 0

        self.not_done = self.not_done * (1 - dones)

        if self.buffer_top != seq_len:
            for env_id, info in enumerate(infos):
                if self.not_done[env_id]:
                    batch_blocks.append(info['blocks'])
                else:
                    batch_blocks.append(np.zeros((8, block_size, block_size, 1)))

            batch_blocks = np.stack(batch_blocks, 0) * 2 - 1
            self.block_buffer[:, self.buffer_top, :, :, :] = batch_blocks
            self.mask_buffer[:, self.buffer_top] = self.not_done

        self.buffer_top += 1

    def close(self):
        self.sess.close()


