from environments.gridworld_vec_env import Grid_World_Vec_Env
from config.gridworld_config import config
from prior_learning.gridworld.train import get_maps
from imitation.gridworld.imitation_model import imitation_policy
import os

import numpy as np
import torch
import os

plan_step = config['plan_step']
num_envs = config['num_envs']
size = config['size']
categories = config['categories']
feature_dim = config['feature_dim']
seq_len = config['seq_len']
follow_steps = config['follow_steps']
save_dir = config['save_dir']

np.random.seed(0)

grid_map_train = np.random.randint(0, categories, size=(size, size))
grid_map_train[:, 0] = 0
grid_map_train[:, -1] = 0
grid_map_train[0, :] = 0
grid_map_train[-1, :] = 0
grid_map_test = np.random.randint(0, categories, size=(size, size))
grid_map_test[:, 0] = 0
grid_map_test[:, -1] = 0
grid_map_test[0, :] = 0
grid_map_test[-1, :] = 0

feature_encoding = np.random.random((categories, feature_dim))
noise_train = np.random.normal(scale=0.05, size=(size, size, feature_dim))
noise_test = np.random.normal(scale=0.05, size=(size, size, feature_dim))

feature_map_train = feature_encoding[grid_map_train.flatten()].reshape((size, size, feature_dim)) + noise_train
feature_map_test = feature_encoding[grid_map_test.flatten()].reshape((size, size, feature_dim)) + noise_test

reward_map_train, net = get_maps(grid_map_train, feature_map_train, feature_encoding, seq_len, categories)
print('====================')
print(grid_map_train)
print('====================')
print(grid_map_test)
print('====================')
reward_map_test = net(torch.from_numpy(feature_map_test.reshape((size * size, feature_dim))).float().cuda()).detach().cpu().numpy().reshape((size, size))

print('====================')
print(np.mean(np.abs(grid_map_train - reward_map_train)), np.mean(np.abs(grid_map_test - reward_map_test)))
print('====================')
"""

######################################################################################
rw_env_train = Grid_World_Vec_Env(1, grid_map_train, reward_map_train)
random_policy = imitation_policy(feature_map_train, 'train')

count = 0
experiment_name = 'rw_train'
log_path = os.path.join(save_dir, experiment_name + '.log')
stat = []
last_obses = rw_env_train.reset()
with open(log_path, "a") as logs:
    logs.write("=" * 5 + "\n")

    while count < 50000:
        count += 1
        action = np.random.randint(4, size=1)
        obses, rewards, dones, infos = rw_env_train.step(action)
        random_policy.update(np.array(last_obses)[:, 0], action)
        last_obses = obses

        if dones[0]:
            # print('restart')
            # print(infos[0]['total_reward'])
            stat.append(infos[0]['total_reward'])

            obs = rw_env_train.reset()
        if count % 20000 == 19999:
            print('[{}] {}'.format(count, infos[0]['total_reward']))

    logs.write(str(np.array(stat).mean() ) + '\n')
####################################################################################
random_policy.train(2)

obses = rw_env_train.reset()

count = 0
experiment_name = 'imi_rw_train'
log_path = os.path.join(save_dir, experiment_name + '.log')

with open(log_path, "a") as logs:
    logs.write("=" * 5 + "\n")

    while True:
        count += 1
        action = random_policy.predict(np.array(obses)[:, 0])
        obses, rewards, dones, infos = rw_env_train.step(np.array([action] * num_envs))

        if dones[0]:
            print('restart')
            print(infos[0]['total_reward'])
            logs.write(str(infos[0]['total_reward']) + '\n')

            break

        print('[{}] {}'.format(count, infos[0]['total_reward']))

######################################################################################
random_policy = imitation_policy(feature_map_test, 'eval')
rw_env_test = Grid_World_Vec_Env(1, grid_map_test, reward_map_test)
obses = rw_env_test.reset()

count = 0
experiment_name = 'imi_rw_test'
log_path = os.path.join(save_dir, experiment_name + '.log')

with open(log_path, "a") as logs:
    logs.write("=" * 5 + "\n")

    while True:
        count += 1
        action = random_policy.predict(np.array(obses)[:, 0])
        obses, rewards, dones, infos = rw_env_test.step(np.array([action] * num_envs))

        if dones[0]:
            print('restart')
            print(infos[0]['total_reward'])
            logs.write(str(infos[0]['total_reward']) + '\n')

            break

        print('[{}] {}'.format(count, infos[0]['total_reward']))

######################################################################################


# count = 0
# experiment_name = 'rw_test'
# log_path = os.path.join(save_dir, experiment_name + '.log')
# stat = []
# last_obses = rw_env_test.reset()
#
# with open(log_path, "a") as logs:
#     logs.write("=" * 5 + "\n")
#
#     while count < 50000:
#         count += 1
#         obses, rewards, dones, infos = rw_env_test.step(np.random.randint(4, size=1))
#
#         if dones[0]:
#             print('restart')
#             print(infos[0]['total_reward'])
#             stat.append(infos[0]['total_reward'])
#             obs = rw_env_test.reset()
#
#         # print('[{}] {}'.format(count, infos[0]['total_reward']))
#
#     logs.write(str(np.array(stat).mean()) + '\n')
"""
######################################################################################
baseline_policy = imitation_policy(feature_map_train, 'train')

env_train = Grid_World_Vec_Env(num_envs, grid_map_train, reward_map_train)

last_obses = env_train.reset()

count = 0
experiment_name = 'ours_train'
log_path = os.path.join(save_dir, experiment_name + '.log')

action_search = np.stack(np.meshgrid(*[[0, 1, 2, 3] for _ in range(plan_step)]), -1).reshape(-1, plan_step).T

with open(log_path, "a") as logs:
    logs.write("=" * 5 + "\n")

    while count < 40:
        # action_trials = np.random.randint(4, size=[plan_step, num_envs])
        action_trials = action_search
        reward_trails = np.zeros((plan_step, num_envs))

        env_train.backup()
        for i, action in enumerate(action_trials):
            obses, rewards, dones, infos = env_train.step(action)
            reward_trails[i, :] = np.array([info['predicted_reward'] for info in infos])

        env_train.restore()
        best_idx = np.argmax(np.sum(reward_trails, 0))
        actions = action_trials[:follow_steps, best_idx]
        for action in actions:
            count += 1
            # print(np.array(last_obses)[:, 0])
            baseline_policy.update(np.array(last_obses)[:, 0], action)
            obses, rewards, dones, infos = env_train.step(np.array([action] * num_envs))
            last_obses = obses
            if dones[0]:
                print('restart')
                print(infos[0]['total_reward'])
                logs.write(str(infos[0]['total_reward']) + '\n')
                obs = env_train.reset()
                break

        print('[{}] {}'.format(count, infos[0]['total_reward']))


######################################################################################
env_test = Grid_World_Vec_Env(num_envs, grid_map_test, reward_map_test)
env_test.reset()

count = 0
experiment_name = 'ours_test'
log_path = os.path.join(save_dir, experiment_name + '.log')

with open(log_path, "a") as logs:
    logs.write("=" * 5 + "\n")

    while count < 40:
        # action_trials = np.random.randint(4, size=[plan_step, num_envs])
        action_trials = action_search
        reward_trails = np.zeros((plan_step, num_envs))

        env_test.backup()
        for i, action in enumerate(action_trials):
            obses, rewards, dones, infos = env_test.step(action)
            reward_trails[i, :] = np.array([info['predicted_reward'] for info in infos])

        env_test.restore()
        best_idx = np.argmax(np.sum(reward_trails, 0))
        actions = action_trials[:follow_steps, best_idx]
        for action in actions:
            count += 1
            obses, rewards, dones, infos = env_test.step(np.array([action] * num_envs))
            if dones[0]:
                print('restart')
                print(infos[0]['total_reward'])
                logs.write(str(infos[0]['total_reward']) + '\n')
                obs = env_test.reset()
                break

        print('[{}] {}'.format(count, infos[0]['total_reward']))

######################################################################################
baseline_policy.train(3)

obses = env_train.reset()

count = 0
experiment_name = 'baseline_train'
log_path = os.path.join(save_dir, experiment_name + '.log')

with open(log_path, "a") as logs:
    logs.write("=" * 5 + "\n")

    while True:
        count += 1
        action = baseline_policy.predict(np.array(obses)[:, 0])
        obses, rewards, dones, infos = env_train.step(np.array([action] * num_envs))

        if dones[0]:
            print('restart')
            print(infos[0]['total_reward'])
            logs.write(str(infos[0]['total_reward']) + '\n')

            break

        print('[{}] {}'.format(count, infos[0]['total_reward']))

######################################################################################
baseline_policy = imitation_policy(feature_map_test, 'eval')
obses = env_test.reset()

count = 0
experiment_name = 'baseline_test'
log_path = os.path.join(save_dir, experiment_name + '.log')

with open(log_path, "a") as logs:
    logs.write("=" * 5 + "\n")

    while True:
        count += 1
        action = baseline_policy.predict(np.array(obses)[:, 0])
        obses, rewards, dones, infos = env_test.step(np.array([action] * num_envs))

        if dones[0]:
            print('restart')
            print(infos[0]['total_reward'])
            logs.write(str(infos[0]['total_reward']) + '\n')

            break

        print('[{}] {}'.format(count, infos[0]['total_reward']))


