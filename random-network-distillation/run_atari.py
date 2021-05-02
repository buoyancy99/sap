#!/usr/bin/env python3
import functools
import os

from stable_baselines import logger
from mpi4py import MPI
import mpi_util
import tf_util
from cmd_util import make_atari_env, arg_parser
from policies.cnn_gru_policy_dynamics import CnnGruPolicy
from policies.cnn_policy_param_matched import CnnPolicy
from ppo_agent import PpoAgent
from utils import set_global_seeds
from vec_env import VecFrameStack
import pprint
import tensorflow as tf
import pickle
import datetime


def train(*, env_id, num_env, hps, num_timesteps, seed, test=False, e_greedy=0, **kwargs):
    # import pdb; pdb.set_trace()
    print('_________________________________________________________')
    pprint.pprint(f'hyperparams: {hps}', width=1)
    pprint.pprint(f'additional hyperparams: {kwargs}', width=1)
    print('_________________________________________________________')
    # TODO: this is just for debugging
    # tmp_env = make_atari_env(env_id, num_env, seed, wrapper_kwargs=dict(clip_rewards=kwargs['clip_rewards']))
    venv = VecFrameStack(make_atari_env(env_id, num_env, seed, wrapper_kwargs=dict(clip_rewards=kwargs['clip_rewards']),
                         start_index=num_env * MPI.COMM_WORLD.Get_rank(),
                         max_episode_steps=hps.pop('max_episode_steps'), action_space=kwargs['action_space']),
                         hps.pop('frame_stack'))
    # venv.score_multiple = {'Mario': 500,
    #                        'MontezumaRevengeNoFrameskip-v4': 100,
    #                        'GravitarNoFrameskip-v4': 250,
    #                        'PrivateEyeNoFrameskip-v4': 500,
    #                        'SolarisNoFrameskip-v4': None,
    #                        'VentureNoFrameskip-v4': 200,
    #                        'PitfallNoFrameskip-v4': 100,
    #                        }[env_id]
    venv.score_multiple = 1   # TODO: understand what is score multiple
    venv.record_obs = True if env_id == 'SolarisNoFrameskip-v4' else False
    ob_space = venv.observation_space
    ac_space = venv.action_space
    gamma = hps.pop('gamma')
    policy = {'rnn': CnnGruPolicy,
              'cnn': CnnPolicy}[hps.pop('policy')]
    agent = PpoAgent(
        scope='ppo',
        ob_space=ob_space,
        ac_space=ac_space,
        stochpol_fn=functools.partial(
            policy,
                scope='pol',
                ob_space=ob_space,
                ac_space=ac_space,
                update_ob_stats_independently_per_gpu=hps.pop('update_ob_stats_independently_per_gpu'),
                proportion_of_exp_used_for_predictor_update=hps.pop('proportion_of_exp_used_for_predictor_update'),
                dynamics_bonus = hps.pop("dynamics_bonus")
            ),
        gamma=gamma,
        gamma_ext=hps.pop('gamma_ext'),
        lam=hps.pop('lam'),
        nepochs=hps.pop('nepochs'),
        nminibatches=hps.pop('nminibatches'),
        lr=hps.pop('lr'),
        cliprange=0.1,
        nsteps=128,
        ent_coef=0.001,
        max_grad_norm=hps.pop('max_grad_norm'),
        use_news=hps.pop("use_news"),
        comm=MPI.COMM_WORLD if MPI.COMM_WORLD.Get_size() > 1 else None,
        update_ob_stats_every_step=hps.pop('update_ob_stats_every_step'),
        int_coeff=hps.pop('int_coeff'),
        ext_coeff=hps.pop('ext_coeff'),
    )
    saver = tf.train.Saver()
    if test:
        agent.restore_model(saver, kwargs['load_dir'], kwargs['exp_name'], mtype=kwargs['load_mtype'])
    agent.start_interaction([venv])

    import time
    st = time.time()
    if hps.pop('update_ob_stats_from_random_agent'):
        if os.path.exists('./data/ob_rms.pkl'):
            with open('./data/ob_rms.pkl', 'rb') as handle:
                agent.stochpol.ob_rms.mean, agent.stochpol.ob_rms.var, agent.stochpol.ob_rms.count = pickle.load(handle)

        else:
            ob_rms = agent.collect_random_statistics(num_timesteps=128*50)   # original number128*50
            with open('./data/ob_rms.pkl', 'wb') as handle:
                pickle.dump([ob_rms.mean, ob_rms.var, ob_rms.count], handle, protocol=2)
    assert len(hps) == 0, "Unused hyperparameters: %s" % list(hps.keys())
    print(f'Time duration {time.time() - st}')

    if not test:
        counter = 1
        while True:
            info = agent.step()
            if info['update']:
                # import pdb; pdb.set_trace()
                logger.logkvs(info['update'])
                logger.dumpkvs()
            if agent.I.stats['tcount'] // 1e7 == counter:
                agent.save_model(saver, kwargs['save_dir'], kwargs['exp_name'], mtype=str(counter*1e7))
                counter += 1
            if agent.I.stats['tcount'] > num_timesteps:
                break
    if test:
        for pkls in range(0, 10):
            print('collecting the', pkls, 'th pickle', ' '*20)
            all_rollout = agent.evaluate(env_id=env_id, seed=seed, episodes=1000, save_image=kwargs['save_image'], e_greedy=e_greedy)
            with open('./data/newworld21_'+str(pkls).zfill(1)+'.pkl', 'wb') as handle:
                pickle.dump(all_rollout, handle)
    agent.stop_interaction()
    if not test:
        agent.save_model(saver, kwargs['save_dir'], kwargs['exp_name'])


def add_env_params(parser):
    parser.add_argument('--env', help='environment ID', default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--max_episode_steps', type=int, default=4500)


def main():
    parser = arg_parser()
    add_env_params(parser)
    parser.add_argument('--num-timesteps', type=int, default=int(1e12))
    parser.add_argument('--num_env', type=int, default=32)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gamma_ext', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--update_ob_stats_every_step', type=int, default=0)
    parser.add_argument('--update_ob_stats_independently_per_gpu', type=int, default=0)
    parser.add_argument('--update_ob_stats_from_random_agent', type=int, default=1)
    parser.add_argument('--proportion_of_exp_used_for_predictor_update', type=float, default=1.)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--policy', type=str, default='rnn', choices=['cnn', 'rnn'])
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--ext_coeff', type=float, default=2.)
    parser.add_argument('--dynamics_bonus', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='/home/hxu/PriorRL/random-network-distillation/ckpts/')
    parser.add_argument('--load_dir', type=str, default='/home/hxu/PriorRL/random-network-distillation/ckpts/')
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--save_image', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='tmp')
    parser.add_argument('--logdir', type=str, default='./logs/')
    parser.add_argument('--clip_rewards', type=int, default=1)
    parser.add_argument('--e_greedy', type=int, default=0)
    parser.add_argument('--action_space', type=str, default='RIGHT_ONLY')
    parser.add_argument('--load_mtype', type=str, default='latest')

    args = parser.parse_args()
    logdir = os.path.join(args.logdir, args.exp_name+'_'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logger.configure(folder=logdir, format_strs=['stdout', 'log', 'csv'] if MPI.COMM_WORLD.Get_rank() == 0 else [])
    if MPI.COMM_WORLD.Get_rank() == 0:
        with open(os.path.join(logger.get_dir(), 'experiment_tag.txt'), 'w') as f:
            f.write(args.tag)
        # shutil.copytree(os.path.dirname(os.path.abspath(__file__)), os.path.join(logger.get_dir(), 'code'))

    mpi_util.setup_mpi_gpus()

    seed = 10000 * args.seed + MPI.COMM_WORLD.Get_rank()
    set_global_seeds(seed)

    hps = dict(
        frame_stack=4,
        nminibatches=4,
        nepochs=4,
        lr=0.0001,
        max_grad_norm=0.0,
        use_news=args.use_news,
        gamma=args.gamma,
        gamma_ext=args.gamma_ext,
        max_episode_steps=args.max_episode_steps,
        lam=args.lam,
        update_ob_stats_every_step=args.update_ob_stats_every_step,
        update_ob_stats_independently_per_gpu=args.update_ob_stats_independently_per_gpu,
        update_ob_stats_from_random_agent=args.update_ob_stats_from_random_agent,
        proportion_of_exp_used_for_predictor_update=args.proportion_of_exp_used_for_predictor_update,
        policy=args.policy,
        int_coeff=args.int_coeff,
        ext_coeff=args.ext_coeff,
        dynamics_bonus = args.dynamics_bonus
    )

    tf_util.make_session(make_default=True)
    train(env_id=args.env, num_env=args.num_env, seed=seed,
          num_timesteps=args.num_timesteps, hps=hps, load_dir=args.load_dir,
          save_dir=args.save_dir, test=args.test, exp_name=args.exp_name,
          clip_rewards=args.clip_rewards, save_image=args.save_image, action_space=args.action_space, e_greedy=args.e_greedy, load_mtype=args.load_mtype)


if __name__ == '__main__':
    main()
