from nes_py.wrappers import JoypadSpace
from environments.gym_super_mario_bros.actions import RIGHT_ONLY
from environments.mario_env import SuperMario_Env
from environments.atari_wrappers import StickyActionEnv, MaxAndSkipEnv, DummyMontezumaInfoWrapper, AddRandomStateToInfo, wrap_deepmind
from config.mario_config import config
from imitation.mario.imitation_model_expert import imitation_model
import os
import numpy as np
import cv2
from tqdm import trange


save_dir= config["save_dir"]
world = config["world"]
stage = config["stage"]
trained_on = config["trained_on"]
save_video = config["save_video"]

experiment_name = 'imitation_expert_world{}_stage{}_trained_on{}'.format(world, stage, trained_on)

log_path = os.path.join(save_dir, experiment_name + '.log')
stat_path = os.path.join(save_dir, experiment_name + '.npy')

if save_video:
    video_path = os.path.join(save_dir, experiment_name + '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowriter = cv2.VideoWriter(video_path, fourcc, 24.0, (512,512))

def get_marioenv(world=1, stage=1, version=0, movement = RIGHT_ONLY, max_episode_steps=4500):
    mario_env = JoypadSpace(SuperMario_Env(world, stage, version), movement)
    mario_env._max_episode_steps = max_episode_steps * 4
    mario_env = StickyActionEnv(mario_env)
    mario_env = MaxAndSkipEnv(mario_env, skip=4)
    mario_env = DummyMontezumaInfoWrapper(mario_env)
    mario_env = AddRandomStateToInfo(mario_env)
    mario_env = wrap_deepmind(mario_env, frame_stack=True)
    return mario_env

def bench_imitation():
    # with open(log_path, "a") as logs:
    #     logs.write("="*5 + "\n")

    env = get_marioenv(world, stage)
    model = imitation_model()
    last_count = 0
    stat = []

    obs = env.reset()

    for count in trange(1, 50001):
        _, action_dist = model.predict(obs[None])
        action = np.random.choice(5, 1, p=action_dist[0])[0]

        obs, _, done, info = env.step(action)
        if save_video:
            videowriter.write(cv2.cvtColor(cv2.resize(np.uint8(info['rgb'] * 255.0), (512, 512), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))

        if done or info['x_pos'] > 3150:
            stat.append(info['x_pos'])
            last_count = count
            # with open(log_path, "a") as logs:
            #     logs.write(str(info['x_pos']) + '\n')
            # print('restart')
            # print(experiment_name)
            obs = env.reset()

        # print('[{}] {}'.format(count, info['x_pos']))

    stat = np.array(stat)
    np.save(stat_path, stat)
    print('mean: {:.3f}, stderr: {:.3f}'.format(stat.mean(), stat.std() / np.sqrt(len(stat))))



if __name__=='__main__':
    bench_imitation()
