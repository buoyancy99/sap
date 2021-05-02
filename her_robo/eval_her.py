import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize
from baselines.common.cmd_util import make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from environments.robo_env.register_envs import register_envs

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = 'her'
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env


def get_env_type(args):
    env_id = args.env
    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def play(args):
    from baselines.common import set_global_seeds
    import pickle
    import her_robo.experiment.config as config
    from her_robo.rollout import RolloutWorker

    set_global_seeds(args.seed)
    # Load policy.
    machine = 'mac'
    if  machine == "mac":
        ppath = args.policy_pickle_path + '_mac'
    else:
        ppath = args.policy_pickle_path
    policy_file = osp.join(ppath, args.exp_name, 'policy_latest.pkl')
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_Q': False,
        'rollout_batch_size': 1,
    }

    for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]

    evaluator = RolloutWorker(params['make_env'](), policy, dims, logger, **eval_params)

    # Run evaluation.
    evaluator.clear_history()
    all_episodes = []
    for i in range(50,100):
        for _ in range(args.n_test_rollouts):
            episode = evaluator.gen_rollouts_render()
            all_episodes.append(episode)
            # import pdb;
            # pdb.set_trace()
            world = evaluator.venv.world_array
            with open(args.save_traj_dir+str(i).zfill(2), 'wb') as f:
                pickle.dump([all_episodes, world], f)


    # record logs
    # for key, val in evaluator.logs('test'):
    #     logger.record_tabular(key, np.mean(val))
    # logger.dump_tabular()



if __name__ == '__main__':
    from her_robo.util import arg_parser
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--env_type',
                        help='type of environment, used when the environment type cannot be automatically determined',
                        type=str)
    parser.add_argument('--seed', help='RNG seed', type=int, default=42)
    parser.add_argument('--n_test_rollouts', type=int, default=500),
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--policy_pickle_path', default='./ckpt', type=str, help="the place to save policy")
    parser.add_argument('--exp_name', default='tmp', type=str, help="experiment name")
    parser.add_argument('--save_traj_dir', default='./data/example',help='the dir to save data', type=str)
    args = parser.parse_args()
    # main(sys.argv)
    register_envs()
    play(args)