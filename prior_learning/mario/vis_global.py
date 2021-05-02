from tensorflow.python import pywrap_tensorflow
import os
import tensorflow as tf
from config.mario_config import config
from skimage.util.shape import view_as_windows
from skimage.transform import rescale
import matplotlib.pyplot as plt

import numpy as np

block_size = config["block_size"]
trained_on = config["trained_on"]
action_space = len(config["movement"])

class Visualizer:
    def __init__(self, stride = 1):
        self.get_model()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.stride = stride
        self.out_dim = 72 // stride + 1

    def get_model(self):
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

        self.input_blocks = tf.placeholder(dtype=tf.float32, shape=[None, block_size, block_size, 1])
        batch_size = tf.cast(tf.shape(self.input_blocks)[0], tf.int64)

        X = tf.nn.relu(tf.nn.conv2d(self.input_blocks, filter=conv1_w, strides=[1, 2, 2, 1], padding="SAME") + conv1_b)
        X = tf.nn.relu(tf.nn.conv2d(X, filter=conv2_w, strides=[1, 2, 2, 1], padding="SAME") + conv2_b)
        X = tf.reshape(X, (-1, 16 * 3 * 3))
        X = tf.nn.relu(tf.matmul(X, fc1_w) + fc1_b)
        X = tf.matmul(X, fc2_w) + fc2_b
        self.reward_map = tf.reshape(X, (batch_size, 8, action_space))

    def inference_single(self, block, pos = None, action = None):
        assert block.shape == (block_size, block_size) or block.shape == (block_size, block_size, 1)
        assert np.max(block) <= 1.0 and np.min(block) >= 0.0
        block = block * 2 - 1
        block = block.reshape((1, block_size, block_size, 1))
        reward_map = self.sess.run(self.reward_map, feed_dict={self.input_blocks: block})[0]
        if pos == None and action ==None:
            return reward_map
        else:
            return reward_map[pos, action]

    def inference_img(self, img, pos = None, action = None):
        assert img.shape == (84, 84, 1) or img.shape == (84, 84)
        assert np.max(img) <= 1.0 and np.min(img) >= 0.0
        img = img.reshape((84, 84))
        img = img * 2 - 1
        img = view_as_windows(img, (12, 12), step=self.stride).reshape((self.out_dim * self.out_dim, 12, 12, 1))
        reward_map = self.sess.run(self.reward_map, feed_dict={self.input_blocks: img})
        reward_map = reward_map.reshape((self.out_dim, self.out_dim, 8 * action_space))
        reward_map = rescale(reward_map, 73 / self.out_dim, order = 0, multichannel=True, preserve_range=True, anti_aliasing = False).reshape(73, 73, 8, action_space)
        if pos == None and action == None:
            return reward_map
        else:
            return reward_map[:, :, pos, action]

        return reward_map

def stack_reward_map(reward_map):
    tl = reward_map[:-18, :-18, 0,:]
    tm = reward_map[:-18, 9:-9, 1,:]
    tr = reward_map[:-18, 18: , 2,:]
    ml = reward_map[9:-9, :-18, 3,:]
    mr = reward_map[9:-9, 18: , 4,:]
    bl = reward_map[18:, :-18, 5,:]
    bm = reward_map[18:, 9:-9, 6,:]
    br = reward_map[18:, 18: , 7,:]
    return tl + tm + tr + ml + mr + bl + bm + br
#[black, d blue, l blue, grey, white]
color_map = np.array([[0x00, 0x00, 0x00], [0xad, 0x95, 0x19], [0xe2, 0xd6, 0xa1], [0xbe, 0xba, 0xbc], [0xf2, 0xf1, 0xf1]], dtype=np.uint8)[None, None]
arrow_map = np.array([[0, 0], [3.5, 0], [2.1, 2.8], [4.0, 0], [2.4, 3.2]], dtype=np.uint8)[None, None]
if __name__ == "__main__":
    import cv2
    from environments.mario_vec_env import SuperMario_Vec_Env
    visualizer = Visualizer()
    env = SuperMario_Vec_Env(1, 1, 2, wrap_atari=True)
    obs, info = env.reset()
    obs = obs[0]
    action = 0
    cv2.namedWindow('vis')

    # create trackbars for color change
    cv2.createTrackbar('pos', 'vis', 0, 7, lambda x:None)
    cv2.createTrackbar('action', 'vis', 0, 4, lambda x:None)

    sticky_left = 0
    while True:
        # time.sleep(0.05)
        obs, rewards, dones, info = env.step(np.array([action]))
        rgb = info[0]['rgb']
        obs = obs[0]
        mario_pos = [min(int(info[0]['screen_x_pos'] * 84 / 256) + 3, 83), min(int((274 - info[0]['y_pos']) * 84 / 240)+1, 83)]

        pos = cv2.getTrackbarPos('pos', 'vis')
        action_to_view = cv2.getTrackbarPos('action', 'vis')
        reward_map = visualizer.inference_img(obs)
        stacked_reward_map = stack_reward_map(reward_map)
        reward_map = np.clip(reward_map, -20, 10)
        reward_map = np.uint8((reward_map + 20) / 30 * 255.0)
        stacked_reward_map = np.clip(stacked_reward_map, -20, 8)
        stacked_reward_map = np.uint8((stacked_reward_map + 20) / 30 * 255.0)
        reward_map_high_contrast = cv2.equalizeHist(reward_map.reshape((73 * 73, 8 * action_space))).reshape((73, 73, 8, action_space))
        stacked_reward_map_high_contrast = stacked_reward_map
        reward_img = cv2.resize(reward_map_high_contrast[:, :, pos, action_to_view], (512, 512), interpolation=cv2.INTER_NEAREST)
        stacked_reward_map_high_contrast = cv2.copyMakeBorder(stacked_reward_map_high_contrast, 15, 14, 15, 14, cv2.BORDER_CONSTANT, (0,))
        stacked_reward_img = cv2.resize(stacked_reward_map_high_contrast[:, :, action_to_view], (512, 512), interpolation=cv2.INTER_NEAREST)
        best_action = (np.argmax(stacked_reward_map_high_contrast, 2)[:, :, None] == np.arange(action_space)[None, None])[:,:,:,None]
        arrow = np.flip(np.sum(best_action * arrow_map, 2), 0)
        fig, ax = plt.subplots(figsize=(20,20))
        arrow = rescale(arrow, 0.5, order=1, multichannel=True, preserve_range=True, anti_aliasing=False)
        ax.imshow(rgb, extent=[-0.5, 84.5, -0.5, 84.5])
        ax.quiver(np.arange(0, 84, 2), np.arange(0, 84, 2), arrow[:,:,0], arrow[:,:,1], pivot='mid', scale = 180)
        plt.savefig('visualization/map.png')
        best_action = np.uint8(np.sum(best_action * color_map, 2))
        best_value = cv2.cvtColor(np.uint8(np.max(stacked_reward_map_high_contrast, 2) * 255.0), cv2.COLOR_GRAY2BGR)
        best_action[mario_pos[1], mario_pos[0]] = np.uint8([0, 255, 0])
        best_action = cv2.resize(best_action, (512, 512), interpolation=cv2.INTER_NEAREST)
        best_value = cv2.resize(best_value, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('action', np.hstack([best_action, best_value]))
        img = cv2.resize(np.uint8(obs * 255.0), (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('vis', np.hstack([img, reward_img, stacked_reward_img]))

        if sticky_left <= 0:
            action = 0

        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break
        if k == ord('e'):
            action = 4
            sticky_left = 5
        elif k == ord('d'):
            action = 3
            sticky_left = 5

        if k == ord('b'):
            env.backup()

        if k == ord('r'):
            env.restore()

        if k == ord('q'):
            break

        sticky_left -= 1



    pass


