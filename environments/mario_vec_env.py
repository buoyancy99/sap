import os
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines import logger
from stable_baselines.bench import Monitor
from nes_py.wrappers import JoypadSpace
from environments.gym_super_mario_bros.actions import RIGHT_ONLY
from environments.mario_env import SuperMario_Env
from environments.atari_wrappers import StickyActionEnv, MaxAndSkipEnv, DummyMontezumaInfoWrapper, AddRandomStateToInfo, wrap_deepmind, BlocksWrapper

class SuperMario_Vec_Env(SubprocVecEnv):
    def __init__(self, num_envs, world=1, stage=1, version=0, movement = RIGHT_ONLY, seed = 0, wrap_atari = False, start_index=0, max_episode_steps=4500):

        def make_mario_env(rank):
            def _thunk():
                mario_env = JoypadSpace(SuperMario_Env(world, stage, version), movement)

                if wrap_atari:
                    mario_env._max_episode_steps = max_episode_steps * 4
                    mario_env = StickyActionEnv(mario_env)
                    mario_env = MaxAndSkipEnv(mario_env, skip=4)
                    mario_env = DummyMontezumaInfoWrapper(mario_env)
                    mario_env = AddRandomStateToInfo(mario_env)
                # mario_env.seed(seed + rank)

                mario_env = Monitor(mario_env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
                if wrap_atari:
                    mario_env = wrap_deepmind(mario_env)
                    mario_env = BlocksWrapper(mario_env)

                mario_env.seed(seed)

                return mario_env


            return _thunk

        self.wrap_atari = wrap_atari
        self.num_envs = num_envs
        super().__init__([make_mario_env(i + start_index) for i in range(num_envs)])


    def backup(self):
        self.env_method('backup', indices=range(self.num_envs))

    def restore(self):
        self.env_method('restore', indices=range(self.num_envs))

    def reset(self):
        obs = super().reset()
        if self.wrap_atari:
            return obs, self.env_method('get_info', indices=range(self.num_envs))
        else:
            return obs, None




if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm
    import cv2
    N = 5 # N parallel env

    env = SuperMario_Vec_Env(N, 1, 1, wrap_atari =True)

    print(env.observation_space) # Box(240, 256, 3)
    print(env.action_space) # Discrete(12)
    obs, info = env.reset()
    # print(obs.shape)


    sync_action = True

    for i in tqdm(range(100000)):
        if sync_action:
            obs, rewards, dones, info = env.step(np.ones(N))
            # obs, rewards, dones, info = env.step(np.array([np.random.randint(env.action_space.n)] * N))
        else:
            obs, rewards, dones, info = env.step(np.random.randint(env.action_space.n, size = N))

        # print(info[0]['screen_x_pos'], info[0]['y_pos'] - 79)
        # env.render()
        # obs[0][274 - info[0]['y_pos'], info[0]['screen_x_pos']] = 255
        # print(info[0]['blocks'].shape)
        cv2.imshow('obs0', cv2.resize( np.uint8(obs[0] * 255.0), (512, 512)))

        # cv2.imshow('obs1', cv2.cvtColor(cv2.resize(obs[1], (512, 512)), cv2.COLOR_BGR2RGB))
        cv2.imshow('blocks', cv2.resize(np.uint8(info[0]['blocks'][4] * 255.0), (512, 512), interpolation=cv2.INTER_NEAREST))
        k = cv2.waitKey(5)

        if k == ord('b'):
            env.backup()

        if k == ord('o'):
            sync_action = True
            env.reset()

        if k == ord('r'):
            env.restore()

        if k == ord('a'):
            sync_action = False

        if k == ord('q'):
            break



        # obs is [N, 240, 256, 3]
        # rewards is [N]
        # dones is [N]
        # infos is a list of dictionaries containing things as described on https://github.com/Kautenja/gym-super-mario-bros