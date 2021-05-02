import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, layers, metrics
import os, math

class FLAGS(object):
    hidden_sizes = [500, 500]

    loss = 'L2'  # possibly L1, L2, MSE, G
    lr = 1e-3
    weight_decay = 1e-5
    batch_size = 128
    max_grad_norm = 2.
    n_epochs = 1000
    n_dev_epochs = 50


class DynamicsModel():
    FLAGS = FLAGS

    def __init__(self, normalizer, dim_state, dim_action, model_path, current_state, goal, mode='train'):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.hidden_sizes = FLAGS.hidden_sizes
        self.model_path = os.path.join(model_path, 'dynamics')
        self.mode = mode
        self.model = self.make_model()
        if mode == 'eval':
            self.model.load_weights(self.model_path)
        self.normalizer = normalizer
        self.current_state = current_state
        self.goal = goal


    def make_model(self):
        if self.mode == 'train':
            dataset = None
            block_input, action_input, pos_input, gt, done = dataset.get_next()
            block_input = self.normalizer(block_input)
            action_input = self.normalizer(action_input)
            block_input = Input(tensor=block_input)  # [None, 18, 18, 1]
            action_input = Input(tensor=action_input)  # [None, action_dim]
        elif self.mode == 'eval':
            block_input = Input(shape=[self.dim_state])
            action_input = Input(shape=[self.dim_action])
        else:
            raise NotImplementedError

        input_features = tf.concat([block_input, action_input], axis=1)
        input_features = layers.Dense(self.hidden_sizes[0], activation='relu')(input_features)
        X = layers.Dense(self.hidden_sizes[1], activation='relu')(input_features)
        state_output = layers.Dense(self.dim_state)(X)
        model = tf.keras.Model(inputs=[block_input, action_input], outputs=[state_output])

        if self.mode == 'train':
            model.compile(optimizer='adam',
                          loss=[tf.losses.mean_squared_error])
        elif self.mode == 'eval':
            model.compile(optimizer='adam',
                          loss=[tf.losses.mean_squared_error])
        return model

    def train(self, epochs=2):
        print(self.model.summary())
        def scheduler(epoch):
            if epoch < 1:
                return 0.0002
            else:
                return 0.0002 * math.exp(-0.3 *  epoch)

        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        self.model.fit(epochs=epochs, steps_per_epoch=200000, callbacks=[callback])
        self.model.save_weights(self.model_path)

    def _get_action_input(self, action):
        action = np.eye(self.dim_action)[action]
        action = action.astype(np.float32)
        return action

    def _get_state_input(self):
        return self.current_state

    def _get_done(self, state, goal):
        return np.linalg.norm(state - goal) < 0.1

    def step(self, actions):
        action_input = self._get_action_input(actions)
        state_input = self._get_state_input()
        pred_state = self.model.predict([state_input, action_input])
        pred_state = self.normalizer(pred_state, inverse=True)


        dones = self._get_done(pred_state, self.goal)

        return pred_state, None, dones.flatten(), {}