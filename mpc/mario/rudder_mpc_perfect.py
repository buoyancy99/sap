from mpc.mario.rudder_reward_model import reward_predictor
from environments.mario_vec_env import SuperMario_Vec_Env
import numpy as np
from config.mario_config import config
import cv2
import os
from tqdm import trange

plan_step = config["plan_step"]
num_envs = config["num_envs"]
follow_steps = config["follow_steps"]
sticky_prob = config["sticky_prob"]
save_dir= config["save_dir"]
reward_type = "mbhp" if config["mbhp"] else "ours"
world = config["world"]
stage = config["stage"]
gamma = config["gamma"]
trained_on = config["trained_on"]
nodeath = config["nodeath"]
nodeath_name = "_nodeath" if nodeath else ""
down_sample = config["down_sample"]
save_video = config["save_video"]

experiment_name = 'rudder_{}_perfect{}_plan{}_follow{}_nenv{}_sticky{}_gamma{}_world{}_stage{}_trained_on{}'.format(reward_type, nodeath_name, plan_step, follow_steps, num_envs,sticky_prob, gamma, world, stage, trained_on)
log_path = os.path.join(save_dir, experiment_name + '.log')
stat_path = os.path.join(save_dir, experiment_name + '.npy')

if save_video:
    video_path = os.path.join(save_dir, experiment_name + '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowriter = cv2.VideoWriter(video_path, fourcc, 24.0, (512,512))

def run_mpc(num_envs):
    value_predictor = reward_predictor()
    env = SuperMario_Vec_Env(num_envs, world, stage, wrap_atari=True)
    obs, info = env.reset()
    env.backup()
    value_predictor.update(info)
    last_count = 0
    stat = []
    last_actions = np.random.randint(env.action_space.n, size=num_envs)
    # with open(log_path, "a") as logs:
    #     logs.write("="*5 + "\n")
    for count in trange(1, 50001):
        for t in range(plan_step):
            sticky_mask = np.random.random(num_envs) < sticky_prob
            new_actions = np.random.randint(env.action_space.n, size=num_envs) * (1 - sticky_mask) + sticky_mask * last_actions
            last_actions = new_actions

            obs, rewards, dones, info = env.step(new_actions)
            if nodeath:
                dones = np.zeros_like(dones)
            if (count + t - last_count) // down_sample == 0:
                value_predictor.update(info, dones)
            # predictor.update(info, np.zeros(num_envs))
            # videowriter.write(cv2.resize(np.uint8(obs[0]), (512, 512), interpolation=cv2.INTER_NEAREST))
            # for i in range(2):
            #     cv2.imshow(str(i), cv2.resize(np.uint8(obs[i]), (512, 512), interpolation=cv2.INTER_NEAREST))
            # k = cv2.waitKey(20)
            # env.render()

        best_action = value_predictor.predict()
        env.restore()
        for action  in best_action:
            obs, rewards, dones, info = env.step(np.array([action] * num_envs))
            if (count - last_count) // down_sample == 0:
                value_predictor.update(info)
            if save_video:
                videowriter.write(cv2.resize(np.uint8(obs[0] * 255.0), (512, 512), interpolation=cv2.INTER_NEAREST))
            if dones[0] or info[0]['x_pos'] > 3150:
                stat.append(info[0]['x_pos'])
                last_count = count
                # with open(log_path, "a") as logs:
                #     logs.write(str(info[0]['x_pos']) + '\n')
                # print('restart')
                # print(experiment_name)
                obs, info = env.reset()
                value_predictor.reset()
                value_predictor.update(info)
                break
        # last_obs = obs
        # assert np.equal(last_obs[0], last_obs[1]).all()
        # cv2.imshow('last1', cv2.resize(np.uint8(last_obs[0]), (512, 512), interpolation=cv2.INTER_NEAREST))
        # cv2.imshow('last2', cv2.resize(np.uint8(last_obs[1]), (512, 512), interpolation=cv2.INTER_NEAREST))
        # print('[{}] {}'.format(count, info[0]['x_pos']))
        env.backup()
        # env.render()
    stat = np.array(stat)
    np.save(stat_path, stat)
    print('mean: {:.3f}, stderr: {:.3f}'.format(stat.mean(), stat.std() / np.sqrt(len(stat))))
    value_predictor.close()

if __name__ == "__main__":
    run_mpc(num_envs)

