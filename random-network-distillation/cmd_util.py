"""
Helpers for scripts like run_atari.py.
"""

import os

import gym
from gym.wrappers import FlattenDictWrapper
from mpi4py import MPI
from stable_baselines import logger
from monitor import Monitor
from atari_wrappers import make_atari, wrap_deepmind
from vec_env import SubprocVecEnv


def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0, max_episode_steps=4500, action_space=None):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id, max_episode_steps=max_episode_steps, action_space=action_space)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk # TODO: change back to _thunk
    # set_global_seeds(seed)
    # x = make_env(0)()
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def make_eval_env(env_id, seed, wrapper_kwargs=None,max_episode_steps=4500):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id, max_episode_steps=max_episode_steps)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    return make_env(0)()


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    return parser
