#!/usr/bin/env bash
set -e

python train_finetune.py \
  --task dehazing \
  --pre_model ./ckpt/pretrained_model.pth \
  --training_in_path ./data/dehazing/train_input \
  --training_gt_path ./data/dehazing/train_gt \
  --eval_in_path ./data/dehazing/val_input \
  --eval_gt_path ./data/dehazing/val_gt \
  --unified_path ./experiments \
  --experiment_name ddl_gef_dehazing \
  --total_iters 500000 \
  --lambda_reg 0.2 \
  --gen_prob 0.1 \
  --lambda_fft 0.1

