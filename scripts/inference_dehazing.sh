#!/usr/bin/env bash
set -e

python inference.py \
  --model_path ./ckpt/best_model.pth \
  --input_path ./data/dehazing/test_input \
  --gt_path ./data/dehazing/test_gt \
  --output_path ./results/dehazing \
  --task dehazing \
  --use_ensemble True \
  --save_images True

