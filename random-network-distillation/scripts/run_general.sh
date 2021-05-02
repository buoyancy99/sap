#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=1 python run_atari.py --gamma_ext 0.999 --policy cnn --env mario --num-timesteps 100000000 --test 0 --save_image 0 --exp_name new_level_all > trash.out &
cd ./scripts/
