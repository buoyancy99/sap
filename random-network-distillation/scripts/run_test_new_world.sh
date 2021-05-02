#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=1 python run_atari.py --gamma_ext 0.999 --policy cnn --env mario --num-timesteps 50000000 --test 1 --save_image 0 --exp_name 'no_ext_sanity' --load_mtype '90000000.0' --ext_coeff 0 --e_greedy 0 --load_dir /shared/hxu/projects/prior_rl_data/ckpts/ &
cd ./scripts/
