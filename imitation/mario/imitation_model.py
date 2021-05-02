import tensorflow as tf
import os
from tensorflow.keras import Input, layers
from imitation.mario.dataloader import imitation_dataset
from config.mario_config import config
import math
import numpy as np

action_space = len(config["movement"])
trained_on = config["trained_on"]

class imitation_model:
    def __init__(self, model_path='imitation/mario/ckpts', mode='eval'):
        self.model_path = os.path.join(model_path, 'imitation_{}'.format(trained_on))
        self.mode = mode
        self.model = self.make_model()
        if mode == 'eval':
            self.model.load_weights(self.model_path)

    def make_model(self):
        if self.mode == 'train':
            dataset = imitation_dataset()
            obs_input, action_gt = dataset.get_next()
            obs_input = Input(tensor=obs_input)
        elif self.mode == 'eval':
            obs_input = Input(shape=[84, 84, 4])

        obs_feature = layers.Conv2D(32, 8, 4, padding='same', activation='relu')(obs_input) # 21
        obs_feature = layers.Conv2D(64, 4, 2, padding='same', activation='relu')(obs_feature)  # 11
        obs_feature = layers.Conv2D(128, 3, 2, padding='same', activation='relu')(obs_feature)  # 6
        obs_feature = layers.Conv2D(256, 3, 2, padding='same', activation='relu')(obs_feature)  # 3
        obs_feature = layers.GlobalMaxPool2D()(obs_feature)
        obs_feature = layers.Flatten()(obs_feature)
        obs_feature = layers.Dense(128, activation='relu')(obs_feature)
        action_out = layers.Dense(action_space, activation='softmax')(obs_feature)

        model = tf.keras.Model(inputs=[obs_input], outputs=[action_out])

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
        print(self.model.summary())
        def scheduler(epoch):
            if epoch < 1:
                return 0.0005
            else:
                return 0.0005 * math.exp(- 0.3 *  epoch)

        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        self.model.fit(epochs=epochs, steps_per_epoch=100000, callbacks=[callback])
        self.model.save_weights(self.model_path)

    def predict(self, obs):
        obs = obs * 2 -1
        actions_dist = self.model.predict(obs)
        actions = np.argmax(actions_dist, 1)

        return actions, actions_dist


if __name__ == "__main__":
    model = imitation_model(model_path = 'imitation/mario/ckpts', mode = 'train')
    model.train()
