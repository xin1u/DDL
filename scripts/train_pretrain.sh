#!/usr/bin/env bash
set -e

python train_pretrain.py \
  --gt_dir ./data/gt_images \
  --save_path ./ckpt \
  --experiment_name ddl_pretrain \
  --diffusion_T 50 \
  --total_iters 100000 \
  --batch_size 16 \
  --crop_size 128 \
  --lr 5e-5

