import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
from copy import copy
from config.mario_config import config

block_size = config["block_size"]
mario_size = config["mario_size"]
screen_H = config["screen_H"]
screen_W = config["screen_W"]

cv2.ocl.setUseOpenCL(False)

def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.float32)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return float(np.sign(reward))

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = np.uint8(frame * 255.0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None] / 255.0

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        rl_common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

# class EvalEnv(gym.Wrapper):
#     def __init__(self, env):
#         """Collect data with original observations (RGB + high res) """
#         gym.Wrapper.__init__(self, env)
#
#     def reset(self, **kwargs):
#         observation = self.env.reset(**kwargs)
#         return self.observation(np.array(observation)), observation
#
#     def step(self, action):
#         observation, reward, done, info = self.env.step(action)
#         return self.observation(np.array(observation)), observation, reward, done, info



class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())
        if done:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode'].update(visited_rooms=copy(self.visited_rooms))
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()

class DummyMontezumaInfoWrapper(gym.Wrapper):

    def __init__(self, env):
        super(DummyMontezumaInfoWrapper, self).__init__(env)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode'].update(pos_count=0,
                                   visited_rooms=set([0]))
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()

class AddRandomStateToInfo(gym.Wrapper):
    def __init__(self, env):
        """Adds the random state to the info field on the first step after reset
        """
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, r, d, info = self.env.step(action)
        if d:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode']['rng_at_episode_start'] = self.rng_at_episode_start
        return ob, r, d, info

    def reset(self, **kwargs):
        self.rng_at_episode_start = copy(self.unwrapped.np_random)
        return self.env.reset(**kwargs)

def wrap_deepmind(env, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    # if evaluation:
    #     env = EvalEnv(env)
    # env = NormalizeObservation(env)
    return env


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def reset(self):
        self.last_action = 0
        return self.env.reset()

    def step(self, action):
        # if self.unwrapped.np_random.uniform() < self.p:
        #     action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        info.update({'action_taken': action})
        info.update({'rgb': obs})

        return obs, reward, done, info

class SparseRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.init_x = None

    def reset(self):
        self.init_x = None
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.init_x is None:
            self.init_x = info['x_pos']
        if done:
            reward = info['x_pos'] - self.init_x #info['score']
        else:
            reward = 0
        return obs, reward, done, info

class BlocksWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.scale_x = 84 / 256
        self.scale_y = 84 / 240

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        info['blocks'], info['local_obs'] = self._get_blocks(obs, info['screen_x_pos'] * self.scale_x + 3, (274 - info['y_pos']) * self.scale_y + 1)
        return obs, reward, done, info

    def get_info(self):
        info = self.env.current_info
        frame = self.env.screen
        info['rgb'] = frame / 255.0
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
            frame = frame[:, :, None]

        frame = frame / 255.0

        info['blocks'], info['local_obs'] = self._get_blocks(frame, info['screen_x_pos'] * self.scale_x + 3, (274 - info['y_pos']) * self.scale_y + 1)

        return info

    def _get_blocks(self, obs, x, y):
        x = int(x)
        y = int(y)
        fov = 2 * block_size + mario_size

        left = x - fov // 2
        offset_left = max(0 - left, 0)
        right = left + fov
        offset_right = min(screen_W - right, 0)
        top = y - fov // 2
        offset_top = max(0 - top, 0)
        bottom = top + fov
        offset_bottom = min(screen_H - bottom, 0)

        channel = obs.shape[-1]
        canvas = np.zeros((fov, fov, channel))
        # print(top, offset_top, bottom, offset_bottom, left, offset_left, right, offset_right)
        # print('=====================================')
        canvas[offset_top: max(fov + offset_bottom, 0), offset_left: max(fov + offset_right, 0)] = \
            obs[top + offset_top: max(bottom + offset_bottom, 0), left + offset_left: max(right + offset_right, 0)]
        x, y = fov // 2, fov // 2
        tl = canvas[:block_size, :block_size]
        tm = canvas[:block_size, x - block_size // 2: x + block_size // 2]
        tr = canvas[:block_size, -block_size:]
        ml = canvas[y - block_size // 2: y + block_size // 2, :block_size]
        mr = canvas[y - block_size // 2: y + block_size // 2, -block_size:]
        bl = canvas[-block_size:, :block_size]
        bm = canvas[-block_size:, x - block_size // 2: x + block_size // 2]
        br = canvas[-block_size:, -block_size:]

        return np.stack([tl, tm, tr, ml, mr, bl, bm, br], axis=0), canvas.squeeze()[None]
