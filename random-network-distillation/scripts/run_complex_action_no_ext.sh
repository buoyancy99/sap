#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=1 python run_atari.py --gamma_ext 0.999 --policy cnn --env mario --num-timesteps 100000000 --test 0 --save_image 0 --exp_name complex_action_no_ext --ext_coeff 0 --action_space COMPLEX_MOVEMENT  &
cd ./scripts/
