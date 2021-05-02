#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=1 python run_atari.py --gamma_ext 0.999 --policy cnn --env mario --num-timesteps 50000000 --test 1 --save_image 0 --exp_name '' --ext_coeff 0 --e_greedy 1
cd ./scripts/
