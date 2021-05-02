from mpc.mario.greedy_reward_model import greedy_reward_predictor
from environments.mario_vec_env import SuperMario_Vec_Env
import numpy as np
from config.mario_config import config
import cv2
import os

plan_step = config["plan_step"]
num_envs = 1
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

experiment_name = 'greedy_{}_perfect{}_plan{}_follow{}_nenv{}_sticky{}_gamma{}_world{}_stage{}_trained_on{}'.format(reward_type, nodeath_name, plan_step, follow_steps, num_envs,sticky_prob, gamma, world, stage, trained_on)
video_path = os.path.join(save_dir, experiment_name + '.avi')
log_path = os.path.join(save_dir, experiment_name + '.log')
stat_path = os.path.join(save_dir, experiment_name + '.npy')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videowriter = cv2.VideoWriter(video_path, fourcc, 24.0, (512,512), 0)

def run_mpc(num_envs):
    value_predictor = greedy_reward_predictor(mbhp=(reward_type=="mbhp"))
    env = SuperMario_Vec_Env(num_envs, world, stage, wrap_atari=True)
    obs, info = env.reset()
    count = 0
    last_count = 0
    if world==2:
        for i in range(5):
            count += 1
            obs, rewards, dones, info = env.step([1])
    best_action = value_predictor.predict(info)

    stat = []
    with open(log_path, "a") as logs:
        logs.write("="*5 + "\n")
    while count < 50000:
        count += 1
        obs, rewards, dones, info = env.step([best_action])
        if dones[0] or info[0]['x_pos'] > 3150:
            stat.append([count - last_count, info[0]['x_pos']])
            last_count = count
            with open(log_path, "a") as logs:
                logs.write(str(info[0]['x_pos']) + '\n')
            print('restart')
            print(experiment_name)
            obs, info = env.reset()
            if world == 2:
                for i in range(5):
                    count += 1
                    obs, rewards, dones, info = env.step([1])
        best_action = value_predictor.predict(info)
        print('[{}] {}'.format(count, info[0]['x_pos']))

    np.save(stat_path, np.array(stat))
    value_predictor.close()

if __name__ == "__main__":
    run_mpc(num_envs)

