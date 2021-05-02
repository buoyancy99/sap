import tensorflow as tf
from tensorflow.keras import Input, layers
import math
import numpy as np
import os

class imitation_policy:
    def __init__(self, feature_map, mode='train'):
        self.feature_map = feature_map
        self.size = self.feature_map.shape[0]
        self.feature_dim = self.feature_map.shape[-1]
        self.pos_rollout = []
        self.action_rollout = []
        self.mode = mode
        self.model_path = os.path.join('imitation', 'gridworld', 'ckpts', 'baseline_imitation')
        if mode == 'eval':
            self.model = self.get_model()
            self.model.load_weights(self.model_path)

    def update(self, obs, action):
        self.pos_rollout.append(obs)
        self.action_rollout.append(action)

    def get_dataset(self):
        pos_data = tf.data.Dataset.from_tensor_slices(np.array(self.pos_rollout, dtype=int))
        action_data = tf.data.Dataset.from_tensor_slices(np.array(self.action_rollout, dtype=int))
        feature_data = tf.data.Dataset.from_tensor_slices(self.feature_map.reshape((1, -1)).astype(np.float32)).repeat()
        dataset = tf.data.Dataset.zip((pos_data, action_data, feature_data))
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(32)
        dataset = dataset.prefetch(1)
        dataset = dataset.make_one_shot_iterator()
        pos_input, action_gt, map_input = dataset.get_next()
        return pos_input, action_gt, map_input

    def get_model(self):
        if self.mode == 'train':
            pos_input, action_gt, grid_input = self.get_dataset()
            pos_input = Input(tensor=pos_input)
            map_input = Input(tensor=grid_input)
        elif self.mode == 'eval':
            pos_input = Input(shape=[2], dtype="int64")
            map_input = Input(shape=[self.size * self.size * self.feature_dim])

        pos_feature = tf.keras.backend.one_hot(
            pos_input,
            self.size
        )

        pos_feature = layers.Flatten()(pos_feature)
        map_feature = layers.Flatten()(map_input)
        pos_feature = layers.Dense(128, activation='relu')(pos_feature)
        map_feature = layers.Dense(128, activation='relu')(map_feature)
        X = layers.concatenate([pos_feature, map_feature])
        X = layers.Dense(128, activation='relu')(X)
        action_output = layers.Dense(4, activation=tf.nn.softmax)(X)

        model = tf.keras.Model(inputs=[pos_input, map_input],
                               outputs=[action_output])

        if self.mode == 'train':
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'],
                          target_tensors=[action_gt]
                          )
        else:
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy']
                          )

        return model

    def train(self, epochs=5):
        self.model = self.get_model()

        def scheduler(epoch):
            if epoch < 2:
                return 0.001
            else:
                return 0.001 * math.exp(- 0.3 * epoch)

        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        self.model.fit(epochs=epochs, steps_per_epoch=10000, callbacks=[callback])
        self.model.save_weights(self.model_path)

    def predict(self, obs):
        if self.mode == 'train':
            self.mode = 'eval'
            self.model = self.get_model()
            self.model.load_weights(self.model_path)
        result = self.model.predict([obs[None], self.feature_map.flatten()[None]])
        return np.argmax(result)


if __name__ == "__main__":
    policy = imitation_policy(np.random.random(size=(16, 16, 32)))
    for _ in range(1000):
        policy.update([0, 0], 0)

    policy.train(1)
    policy = imitation_policy(np.random.random(size=(16, 16, 32)), 'eval')
    policy.predict(np.array([0, 0]))