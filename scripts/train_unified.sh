#!/usr/bin/env bash
set -e

python train_unified.py \
  --pre_model ./ckpt/pretrained_model.pth \
  --data_root ./data \
  --tasks noisy,rainy,jpeg,snowy,inpainting,raindrop,shadowed,lowlight,hazy,blurry \
  --unified_path ./experiments \
  --experiment_name ddl_unified \
  --iters_per_task 100000 \
  --num_experts 10 \
  --lambda_reg 0.2 \
  --gen_prob 0.1 \
  --lambda_fft 0.1

