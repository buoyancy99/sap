from pathlib import Path

config = {
    # global
    "size": 8,
    # prior learning stage
    "seq_len": 32,  # length of episode used in training. Longer will be skipped and shorter ones padded
    "feature_dim": 16,
    "categories": 16,

    # mpc & dynamics stage
    "plan_step": 4,  # look ahead steps in mpc
    "num_envs": 256,
    "follow_steps": 1,  # actual steps to follow planning
    "mbhp": 0,  # is this using human prior reward (MBHP)
    "save_dir": "./experiments/gridworld",
    'save_video': False
}

Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
