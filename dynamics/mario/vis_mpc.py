from mpc.mario.reward_model import reward_predictor
from environments.mario_vec_env import SuperMario_Vec_Env
import numpy as np
from config.mario_config import config
from dynamics.mario.dynamics_model import dynamics_model
import cv2
import os

screen_H = config["screen_H"]
screen_W = config["screen_W"]
plan_step = config["plan_step"]
num_envs = config["num_envs"]
follow_steps = config["follow_steps"]
sticky_prob = config["sticky_prob"]
baseline = config["baseline"]
save_dir= config["save_dir"]
baseline_name = "baseline" if baseline else "ours"
world = config["world"]
stage = config["stage"]


scale_x = screen_W / 256
scale_y = screen_H / 240

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videowriter = cv2.VideoWriter('debug.avi', fourcc, 12.0, (2048,1024))

def run_mpc(num_envs):
    value_predictor = reward_predictor(baseline=baseline)
    dynamics = dynamics_model(model_path = 'dynamics/mario/ckpts', mode = 'eval', num_envs = num_envs)
    env = SuperMario_Vec_Env(num_envs, world, stage, wrap_atari=True)
    last_obs, info = env.reset()
    last_frame = info[0]['rgb']
    env.backup()
    value_predictor.update(info)
    dynamics.reset(last_obs, info)
    count = 0
    last_actions = np.random.randint(env.action_space.n, size=num_envs)
    while count < 50000:
        count += 1
        dynamics.start_planning()
        for _ in range(plan_step):
            sticky_mask = np.random.random(num_envs) < sticky_prob
            new_actions = np.random.randint(env.action_space.n, size=num_envs) * (1 - sticky_mask) + sticky_mask * last_actions
            last_actions = new_actions

            _, _, dones, info_hat = dynamics.step(new_actions)
            obs, _, _, info = env.step(new_actions)
            value_predictor.update(info_hat, dones)
            y, x = int(info_hat[0]['y'] / scale_y) + 6, int(info_hat[0]['x'] / scale_x) + 7
            last_frame[max(0, min(239, y-2)) :max(0, min(239, y+2)), max(0, min(255, x-2)):max(0, min(255, x+2))] = 0
            obs_to_show = info[0]['rgb']
            y, x = int(274 - info[0]['y_pos']) + 6, int(info[0]['screen_x_pos']) + 7
            obs_to_show[max(0, min(239, y-2)) :max(0, min(239, y+2)), max(0, min(255, x-2)):max(0, min(255, x+2))] = 0

            frame_last = cv2.cvtColor(cv2.resize(np.uint8(last_frame * 255.0), (1024, 1024), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
            frame = cv2.cvtColor(cv2.resize(np.uint8(obs_to_show * 255.0), (1024, 1024), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
            videowriter.write(np.concatenate([frame_last, frame], 1))

            # cv2.imshow('last_obs', frame_last)
            # cv2.imshow('obs', frame)
            # k = cv2.waitKey(5)

        dynamics.end_planning()
        best_action = value_predictor.predict()
        env.restore()
        for action  in best_action:
            obs, rewards, dones, info = env.step(np.array([action] * num_envs))
            dynamics.update(obs, info)
            if dones[0] or info[0]['x_pos'] > 3150:
                print('restart')
                obs, info = env.reset()
                dynamics.reset(obs, info)
                break
        last_frame = info[0]['rgb']
        # assert np.equal(last_obs[0], last_obs[1]).all()
        # cv2.imshow('last1', cv2.resize(np.uint8(last_obs[0]), (512, 512), interpolation=cv2.INTER_NEAREST))
        # cv2.imshow('last2', cv2.resize(np.uint8(last_obs[1]), (512, 512), interpolation=cv2.INTER_NEAREST))
        print('[{}] {}'.format(info[0]['x_pos'], count))
        env.backup()
        value_predictor.update(info)
        # env.render()
    value_predictor.close()

if __name__ == "__main__":
    run_mpc(num_envs)

