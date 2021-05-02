#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=3 python run_atari.py --gamma_ext 0.999 --policy cnn --env mario_sparse --num-timesteps 100000000 --test 0 --save_image 0 --exp_name sparse_11 > trash.out &
cd ./scripts/
