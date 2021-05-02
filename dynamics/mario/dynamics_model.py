import tensorflow as tf
import os
from tensorflow.keras import Input, layers, metrics
import numpy as np
import math
from dynamics.mario.dataloader import dynamics_dataset
from config.mario_config import config

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
gpu_config.log_device_placement = True 
sess = tf.Session(config=gpu_config)

block_size = config["block_size"]
mario_size = config["mario_size"]
screen_H = config["screen_H"]
screen_W = config["screen_W"]
seq_len = config["seq_len"]
action_space = len(config["movement"])
known_len = config["known_len"]
plan_step = config["plan_step"]
v_mean = config["v_mean"]
v_std = config["v_std"]
pos_mean = config["pos_mean"]
pos_std = config["pos_std"]
trained_on = config["trained_on"]

scale_x = screen_W / 256
scale_y = screen_H / 240

class dynamics_model:
    def __init__(self, model_path = 'dynamics/mario/ckpts', mode = 'eval', num_envs = 128):
        self.model_path = os.path.join(model_path, 'dynamics_{}_{}'.format(plan_step, trained_on))
        self.mode = mode
        self.model = self.make_model()
        if mode == 'eval':
            self.model.load_weights(self.model_path)
        self.num_envs = num_envs
        self.isplanning = False
        self.block_input = None

    def initialize_buffer(self):
        self.buffer = np.zeros([self.num_envs, known_len, 3], dtype=np.float32)
        self.action_buffer = np.zeros([self.num_envs, known_len], dtype=np.int64)

    def make_model(self):
        if self.mode == 'train':
            dataset = dynamics_dataset()
            block_input, action_input, pos_input, step_input, gt, done  = dataset.get_next()
            block_input = Input(tensor=block_input)  # [None, 42, 42, 1]
            action_input = Input(tensor=action_input)  # [None, 4]
            pos_input = Input(tensor=pos_input)  # [None, known_len * 2]
            step_input = Input(tensor=step_input)
        elif self.mode == 'eval':
            block_input = Input(shape=[30, 30, 1])
            action_input = Input(shape=[4], dtype="int64")
            pos_input = Input(shape=[(known_len - 1) * 2])
            step_input = Input(shape=[], dtype="int64")

        block_feature = layers.Conv2D(8, 3, 2, padding='same', activation='relu')(block_input) #15
        block_feature = layers.Conv2D(16, 3, 2, padding='same', activation='relu')(block_feature) #8
        block_feature = layers.Conv2D(32, 3, 2, padding='same', activation='relu')(block_feature) #4
        block_feature = layers.Conv2D(64, 3, 2, padding='same', activation='relu')(block_feature) #2
        block_feature = layers.GlobalMaxPool2D()(block_feature)
        block_feature = layers.Flatten()(block_feature)
        step_feature = tf.keras.backend.one_hot(step_input, plan_step)
        action_feature= tf.keras.backend.one_hot(action_input, action_space)
        action_feature = layers.Reshape((action_space * 4, ))(action_feature)
        action_feature = layers.Dense(64, activation='relu')(action_feature)
        pos_feature = layers.Dense(64, activation='relu')(pos_input)
        step_feature = layers.Dense(64, activation='relu')(step_feature)
        X = layers.concatenate([block_feature, action_feature, pos_feature, step_feature])
        X = layers.Dense(128, activation='relu')(X)
        pos_output = layers.Dense(2)(X)
        done_output = layers.Dense(1)(X)
        model = tf.keras.Model(inputs=[block_input, action_input, pos_input, step_input], outputs=[pos_output, done_output])

        if self.mode == 'train':
            model.compile(optimizer='adam',
                          loss=[tf.losses.mean_squared_error, tf.losses.sigmoid_cross_entropy],
                          metrics=[['mse'], [tf.keras.metrics.BinaryAccuracy(0)]],
                          loss_weights = [1, 1],
                          target_tensors=[gt, done]
                          )
        else:
            model.compile(optimizer='adam',
                          loss=[tf.losses.mean_squared_error, tf.losses.sigmoid_cross_entropy],
                          loss_weights=[1, 1],
                          metrics=[['mse'], [tf.keras.metrics.BinaryAccuracy(0)]],
                          )

        return model

    def train(self, epochs=2):
        print(self.model.summary())
        def scheduler(epoch):
            if epoch < 1:
                return 0.0003
            else:
                return 0.0002 * math.exp(- 0.3 *  epoch)

        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        self.model.fit(epochs=epochs, steps_per_epoch=200000, callbacks=[callback])
        self.model.save_weights(self.model_path)

    def update(self, obses, infos):
        self.buffer[:, :-1,:] = self.buffer[:, 1:,:]
        self.cur_obses = obses
        self.cur_screen_x_pos = np.array([info['screen_x_pos'] * scale_x for info in infos])
        self.cur_y_pos = np.array([(274 - info['y_pos']) * scale_y for info in infos])
        self.buffer[:,-1,0] = np.array([info['x_pos'] * scale_x for info in infos])
        self.buffer[:,-1,1] = self.cur_y_pos
        self.buffer[:,-1,2] = self.cur_screen_x_pos

        self.action_buffer[:, :-1] = self.action_buffer[:, 1:]
        self.action_buffer[:, -1] =  np.array([info['action_taken'] for info in infos], dtype=np.int64)

    def _get_pos_input(self):
        v_input = self.buffer[:, 1:, :2] - self.buffer[:, :-1, :2]
        v_input = (v_input - v_mean[None, None, :]) / v_std[None, None, :]
        # posx_input = self.buffer[:, 1:, 2]
        # posy_input = self.buffer[:, 1:, 1]
        # pos_input = np.stack([posx_input, posy_input], 2)
        # pos_input = (pos_input - pos_mean[None, None, :]) / pos_std[None, None, :]
        # pos_input = np.concatenate([pos_input, v_input], 2)
        pos_input = v_input.reshape((self.num_envs, -1))

        return pos_input

    def _get_block_input(self):
        blocks_stacks = []
        blocks_inputs = []
        local_obs = []

        for obs, x, y in zip(self.cur_obses, self.cur_screen_x_pos, self.cur_y_pos):
            x = int(x + 3)
            y = int(y + 1)
            fov = 2 * block_size + mario_size
            x = max(min(x, screen_W + fov), - fov)
            y = max(min(y, screen_H + fov), - fov)

            left = x - fov // 2
            offset_left = max(0 - left, 0)
            right = left + fov
            offset_right = min(screen_W - right, 0)
            top = y - fov // 2
            offset_top = max(0 - top, 0)
            bottom = top + fov
            offset_bottom = min(screen_H - bottom, 0)

            canvas = np.zeros((fov, fov, 1))
            # print(top, offset_top, bottom, offset_bottom, left, offset_left, right, offset_right)
            # print('=====================================')
            canvas[offset_top: max(fov + offset_bottom, 0),
            offset_left: max(fov + offset_right, 0)] = \
                obs[top + offset_top: max(bottom + offset_bottom, 0), left + offset_left:max(right + offset_right, 0)]
            tl = canvas[:block_size, :block_size]
            tm = canvas[:block_size, fov // 2 - block_size // 2: fov // 2 + block_size // 2]
            tr = canvas[:block_size, -block_size:]
            ml = canvas[fov // 2 - block_size // 2: fov // 2 + block_size // 2, :block_size]
            mr = canvas[fov // 2 - block_size // 2: fov // 2 + block_size // 2, -block_size:]
            bl = canvas[-block_size:, :block_size]
            bm = canvas[-block_size:, fov // 2 - block_size // 2: fov // 2 + block_size // 2]
            br = canvas[-block_size:, -block_size:]
            blocks_stacks.append(np.stack([tl, tm, tr, ml, mr, bl, bm, br], axis=0))
            local_obs.append(canvas[:, :, 0][None])

            x = max(min(x, screen_W + mario_size * 5), - mario_size * 5)
            y = max(min(y, screen_H + mario_size * 5), - mario_size * 5)
            left = int(x - mario_size * 5 // 2)
            offset_left = max(0 - left, 0)
            right = left + 5 * mario_size
            offset_right = min(screen_W - right, 0)
            top = int(y - mario_size * 5 // 2)
            offset_top = max(0 - top, 0)
            bottom = top + 5 * mario_size
            offset_bottom = min(screen_H - bottom, 0)
            canvas = np.zeros((5 * mario_size, 5 * mario_size, 1))
            # print(top, offset_top, bottom, offset_bottom, left, offset_left, right, offset_right)
            # print('=====================================')
            canvas[offset_top: max(5 * mario_size + offset_bottom, 0),
            offset_left: max(5 * mario_size + offset_right, 0)] = \
                obs[top + offset_top: max(bottom + offset_bottom, 0), left + offset_left:max(right + offset_right, 0)]

            blocks_inputs.append(canvas)

        blocks_stacks = np.stack(blocks_stacks, 0)
        blocks_inputs = np.stack(blocks_inputs, 0) * 2 - 1
        local_obs = np.stack(local_obs, 0)

        return blocks_stacks, blocks_inputs, local_obs

    def _get_action_input(self, action):
        action = action.reshape((self.num_envs, 1))
        return np.concatenate([self.action_buffer[:, 1:], action], 1)

    def start_planning(self):
        self.isplanning = True
        self.cur_obses_backup = np.copy(self.cur_obses)
        self.cur_screen_x_pos_backup = np.copy(self.cur_screen_x_pos)
        self.cur_y_pos_backup = np.copy(self.cur_y_pos)
        self.buffer_backup = np.copy(self.buffer)
        self.action_buffer_backup = np.copy(self.action_buffer)
        _, self.block_input, _ = self._get_block_input()
        self.step_idx = 0

    def end_planning(self):
        self.isplanning = False
        self.cur_obses = self.cur_obses_backup
        self.cur_screen_x_pos = self.cur_screen_x_pos_backup
        self.cur_y_pos = self.cur_y_pos_backup
        self.buffer = self.buffer_backup
        self.action_buffer = self.action_buffer_backup

    def reset(self, obses, infos):
        self.initialize_buffer()
        self.cur_obses = obses
        self.cur_screen_x_pos = np.array([info['screen_x_pos'] * scale_x for info in infos])
        self.cur_y_pos = np.array([(274 - info['y_pos']) * scale_y for info in infos])
        self.buffer[:, :, 0] = np.array([info['x_pos'] * scale_x for info in infos])[:, None]
        self.buffer[:, :, 1] = self.cur_y_pos[:, None]
        self.buffer[:, :, 2] = self.cur_screen_x_pos[:, None]

    def step(self, actions):
        action_input = self._get_action_input(actions)
        pos_input = self._get_pos_input()
        step_input = np.array([self.step_idx] * self.num_envs, dtype=np.int64)

        d_pos, dones = self.model.predict([self.block_input, action_input, pos_input, step_input])
        d_pos = d_pos * v_std[None, :] + v_mean[None, :]
        # print(d_pos[0, :], '=' * 20)
        self.cur_screen_x_pos = self.cur_screen_x_pos + d_pos[:, 0]
        self.cur_y_pos = self.cur_y_pos + d_pos[:, 1]
        self.action_buffer[:, :-1] = self.action_buffer[:, 1:]
        self.action_buffer[:, -1] = actions
        self.buffer[:, :-1, :] = self.buffer[:, 1:, :]
        self.buffer[:, -1, 0] = self.buffer[:, -1, 0] + d_pos[:, 0]
        self.buffer[:, -1, 1] = self.buffer[:, -1, 1] + d_pos[:, 1]
        self.buffer[:, -1, 2] = self.buffer[:, -1, 2] + d_pos[:, 0]
        self.step_idx += 1

        blocks_stacks, self.block_input, local_obses = self._get_block_input()
        infos = [{} for _ in range(self.num_envs)]
        for info, x, y, action, blocks_stack, local_obs in zip(infos, self.cur_screen_x_pos, self.cur_y_pos, actions, blocks_stacks, local_obses):
            info['x'] = x
            info['y'] = y
            info['action_taken'] = action
            info['blocks'] = blocks_stack
            info['local_obs'] = local_obs

        #if dones[0] > 0:
        #    print('env 0 done {}'.format(dones[0]))
        
        # dones = dones > 0
        dones = 1/(1 + np.exp(-dones))

        return None, None, dones.flatten(), infos


if __name__ == "__main__":
    model = dynamics_model(model_path = 'dynamics/mario/ckpts', mode = 'train', num_envs = 128)
    model.train()
