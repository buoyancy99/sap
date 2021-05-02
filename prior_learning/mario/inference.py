from config.mario_config import config
from tensorflow.python import pywrap_tensorflow
import os
import numpy as np
import tensorflow as tf
import cv2

block_size = config["block_size"]
trained_on = config["trained_on"]
action_space = len(config["movement"])

def print_table(name, reward_mat):
    reward_mat = np.flip(reward_mat, 0)
    print(("[{:^6.2f}|{:^6.2f}|{:^6.2f}|{:^6.2f}|{:^6.2f}] "*3).format(*reward_mat[:3].flatten()))
    print("[{:^6.2f}|{:^6.2f}|{:^6.2f}|{:^6.2f}|{:^6.2f}] [{:^34s}] [{:^6.2f}|{:^6.2f}|{:^6.2f}|{:^6.2f}|{:^6.2f}]".format(*reward_mat[3].flatten(), name, *reward_mat[4].flatten()))
    print(("[{:^6.2f}|{:^6.2f}|{:^6.2f}|{:^6.2f}|{:^6.2f}] "*3).format(*reward_mat[5:].flatten()))
    print("===============================================")

def vis_table(name, reward_mat):
    reward_mat = np.flip(reward_mat, 0)
    block_image = cv2.imread(os.path.join('prior_learning', 'mario', 'blocks', name + '.png'))
    block_image = cv2.resize(block_image, (36, 36))
    for action in range(action_space):
        canvas = np.zeros((72, 72, 3), dtype=np.uint8)
        canvas[:, :, 0] = 0xa4
        canvas[:, :, 1] = 0x96
        canvas[:, :, 2] = 0xec

        canvas[18:54, 18:54] = block_image

        canvas[:18,:18] = reward_mat[0, action]
        canvas[:18,27:45] =reward_mat[1, action]
        canvas[:18,-18:] = reward_mat[2, action]

        canvas[27:45,:18] = reward_mat[3, action]
        canvas[27:45,-18:] = reward_mat[4, action]

        canvas[-18:,:18] = reward_mat[5, action]
        canvas[-18:,27:45] = reward_mat[6, action]
        canvas[-18:,-18:] = reward_mat[7, action]

        cv2.imwrite(os.path.join('visualization', '{}_{}.png'.format(name, action)), canvas)
        print(os.path.join('visualization', '{}_{}.png'.format(name, action)))

def adjust_contrast(input):
    min_pixel = np.min(input)
    max_pixel = np.max(input)
    LUT = np.zeros(256, dtype=np.uint8)
    LUT[min_pixel:max_pixel + 1] = np.linspace(start=0, stop=255, num=(max_pixel - min_pixel) + 1, endpoint=True, dtype=np.uint8)
    return LUT[input]

object_names =["goomba", "koopa", "goomba2", "koopa2", "air", "tube", "ground", "cliff1"]
object_mats = np.stack([np.load(os.path.join('prior_learning', 'mario', 'blocks', n + '.npy')) for n in object_names]).reshape((len(object_names), block_size, block_size, 1))

checkpoint_path = os.path.join('prior_learning', 'mario', 'ckpts', 'reward_{}'.format(trained_on))
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    print("tensor_name: ", key)

conv1_w = tf.convert_to_tensor(reader.get_tensor('conv1/w'))
conv1_b = tf.convert_to_tensor(reader.get_tensor('conv1/b'))
conv2_w = tf.convert_to_tensor(reader.get_tensor('conv2/w'))
conv2_b = tf.convert_to_tensor(reader.get_tensor('conv2/b'))
fc1_w = tf.convert_to_tensor(reader.get_tensor('fc1/w'))
fc1_b = tf.convert_to_tensor(reader.get_tensor('fc1/b'))
fc2_w = tf.convert_to_tensor(reader.get_tensor('fc2/w'))
fc2_b = tf.convert_to_tensor(reader.get_tensor('fc2/b'))

input_block = tf.placeholder(dtype=tf.float32, shape=[None, block_size, block_size, 1])
X = tf.nn.relu(tf.nn.conv2d(input_block, filter=conv1_w, strides=[1, 2, 2, 1], padding="SAME") + conv1_b)
X = tf.nn.relu(tf.nn.conv2d(X, filter=conv2_w, strides=[1, 2, 2, 1], padding="SAME") + conv2_b)
X = tf.reshape(X, (-1, 16 * 3 * 3))
X = tf.nn.relu(tf.matmul(X, fc1_w) + fc1_b)
X = tf.matmul(X, fc2_w) + fc2_b
X = tf.reshape(X, (-1, 8, action_space))

with tf.Session() as sess:
    block_values = sess.run(X, feed_dict={input_block: object_mats * 2 - 1})

    for i, n in enumerate(object_names):
        print_table(n, block_values[i])

    block_values = np.clip(block_values, -1.1, None)
    block_values = (block_values - np.min(block_values)) /  (np.max(block_values) - np.min(block_values))
    block_values = np.uint8(block_values * 255.0)
    block_values = adjust_contrast(block_values)

    for i, n in enumerate(object_names):
        vis_table(n, block_values[i])



