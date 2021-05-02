#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=0 python run_her.py --env=FetchBlocked-v1 --num_timesteps=2.5e6 --save_path=fetchBlocked_1e6 --exp_name=demo > blocked_try2.out &
cd ./scripts/
