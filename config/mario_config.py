from environments.gym_super_mario_bros.actions import RIGHT_ONLY
import numpy as np
from pathlib import Path
from config.parse import get_args

args = get_args()

config = {
    # global
    "block_size": 12,
    "mario_size": 6,
    "screen_H": 84,
    "screen_W": 84,
    "down_sample": 2,
    "skip": 4,  # frame skip in atari env
    "movement": RIGHT_ONLY,
    "trained_on": "1-1",

    # prior learning stage
    "seq_len": 512,  # length of episode used in training. Longer will be skipped and shorter ones padded

    # mpc & dynamics stage
    "world": args.world,
    "stage": 1,
    "nodeath": args.nodeath,
    "plan_step": args.plan_step,  # look ahead steps in mpc
    "known_len": 4,  # length of info history used to predict dynamics
    "num_envs": 128,
    "gamma": 0.95,
    "mbhp": args.mbhp,  # is this using human prior reward (MBHP)
    "follow_steps": 1,  # actual steps to follow planning
    "sticky_prob": 0.2,  # probability of sticky action in mpc
    "save_dir": "./experiments/mario",
    "v_mean": np.array([1.6189826843581956, 0.021213388703836673]),  # mean velocity
    "v_std": np.array([1.2330620697047778, 3.8837359444631727]),
    "pos_mean": np.array([37.96412864593538, 58.20639857691254]),  # mean of screen position
    "pos_std": np.array([3.216992632470916, 9.97151715409656]),
    'save_video': args.store_video
}

Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
