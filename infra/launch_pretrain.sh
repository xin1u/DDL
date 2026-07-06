#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source infra/distributed.env

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

GT_DIR=${GT_DIR:-./data/gt_images}
SAVE_PATH=${SAVE_PATH:-./ckpt}
WRITER_DIR=${WRITER_DIR:-./tf-logs}
EXP_NAME=${EXP_NAME:-ddl_pretrain_ddp}
TOTAL_ITERS=${TOTAL_ITERS:-100000}
BATCH_SIZE=${BATCH_SIZE:-16}
CROP_SIZE=${CROP_SIZE:-128}
LR=${LR:-5e-5}
DIFFUSION_T=${DIFFUSION_T:-50}

if [ "${NNODES}" = "1" ]; then
  torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" train_pretrain.py \
    --gt_dir "${GT_DIR}" \
    --save_path "${SAVE_PATH}" \
    --writer_dir "${WRITER_DIR}" \
    --experiment_name "${EXP_NAME}" \
    --diffusion_T "${DIFFUSION_T}" \
    --total_iters "${TOTAL_ITERS}" \
    --batch_size "${BATCH_SIZE}" \
    --crop_size "${CROP_SIZE}" \
    --lr "${LR}"
else
  torchrun --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --nproc_per_node="${GPUS_PER_NODE}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" train_pretrain.py \
    --gt_dir "${GT_DIR}" \
    --save_path "${SAVE_PATH}" \
    --writer_dir "${WRITER_DIR}" \
    --experiment_name "${EXP_NAME}" \
    --diffusion_T "${DIFFUSION_T}" \
    --total_iters "${TOTAL_ITERS}" \
    --batch_size "${BATCH_SIZE}" \
    --crop_size "${CROP_SIZE}" \
    --lr "${LR}"
fi
