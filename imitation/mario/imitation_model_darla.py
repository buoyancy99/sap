import tensorflow as tf
import os
from tensorflow.keras import Input, layers
from imitation.mario.dataloader_darla import imitation_dataset
from config.mario_config import config
import math
import numpy as np
from DARLA.mario.beta_vae.model import Model as Beta_VAE_Net
import torch

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
gpu_config.log_device_placement = True
sess = tf.Session(config=gpu_config)

action_space = len(config["movement"])
trained_on = config["trained_on"]

class imitation_model:
    def __init__(self, model_path='imitation/mario/ckpts', mode='eval'):
        self.model_path = os.path.join(model_path, 'imitation_{}_darla'.format(trained_on))
        self.mode = mode
        self.model = self.make_model()
        if mode == 'eval':
            print('loading trained model')
            self.model.load_weights(self.model_path)
            self.vae = Beta_VAE_Net().cuda()
            self.vae.load_state_dict(torch.load('DARLA/mario/beta_vae/ckpts/latest.model')['model_state_dict'])
            for param in self.vae.parameters():
                param.requires_grad = False
            print('model loaded')

    def make_model(self):
        if self.mode == 'train':
            dataset = imitation_dataset('{}/Datasets/MarioImitationDarla'.format(os.getenv('DATASET_ROOT')))
            obs_input, action_gt = dataset.get_next()
            obs_input = Input(tensor=obs_input)
        elif self.mode == 'eval':
            obs_input = Input(shape=[128])

        obs_feature = layers.Dense(128, activation='relu')(obs_input)
        obs_feature = layers.Dense(64, activation='relu')(obs_feature)
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

    def train(self, epochs=2):
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
        obs = torch.from_numpy(obs[:, :, :, 1:]).float().cuda()
        obs = obs.permute(0, 3, 1, 2)
        obs = torch.nn.functional.interpolate(obs, size=(64, 64), mode='bilinear')
        encoding = self.vae.encode(obs).detach().cpu().numpy()
        actions_dist = self.model.predict(encoding)
        actions = np.argmax(actions_dist, 1)

        return actions, actions_dist


if __name__ == "__main__":
    model = imitation_model(model_path = 'imitation/mario/ckpts', mode = 'train')
    model.train()
