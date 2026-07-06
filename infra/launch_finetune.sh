#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source infra/distributed.env

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

TASK=${TASK:-dehazing}
PRE_MODEL=${PRE_MODEL:-./ckpt/pretrained_model.pth}
TRAIN_IN=${TRAIN_IN:-./data/dehazing/train_input}
TRAIN_GT=${TRAIN_GT:-./data/dehazing/train_gt}
EVAL_IN=${EVAL_IN:-./data/dehazing/val_input}
EVAL_GT=${EVAL_GT:-./data/dehazing/val_gt}
OUT_DIR=${OUT_DIR:-./experiments}
WRITER_DIR=${WRITER_DIR:-./tf-logs}
EXP_NAME=${EXP_NAME:-ddl_gef_ddp}
TOTAL_ITERS=${TOTAL_ITERS:-500000}
BATCH_SIZE=${BATCH_SIZE:-16}
CROP_SIZE=${CROP_SIZE:-128}
LR=${LR:-5e-5}
PRINT_FREQ=${PRINT_FREQ:-200}
VAL_FREQ=${VAL_FREQ:-5000}
SAVE_FREQ=${SAVE_FREQ:-10000}
DIFFUSION_T=${DIFFUSION_T:-50}
LAMBDA_REG=${LAMBDA_REG:-0.2}
GEN_PROB=${GEN_PROB:-0.1}
IMPORTANCE_BATCHES=${IMPORTANCE_BATCHES:-50}
LAMBDA_FFT=${LAMBDA_FFT:-0.1}

if [ "${NNODES}" = "1" ]; then
  torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" train_finetune.py \
    --task "${TASK}" \
    --pre_model "${PRE_MODEL}" \
    --training_in_path "${TRAIN_IN}" \
    --training_gt_path "${TRAIN_GT}" \
    --eval_in_path "${EVAL_IN}" \
    --eval_gt_path "${EVAL_GT}" \
    --unified_path "${OUT_DIR}" \
    --writer_dir "${WRITER_DIR}" \
    --experiment_name "${EXP_NAME}" \
    --total_iters "${TOTAL_ITERS}" \
    --BATCH_SIZE "${BATCH_SIZE}" \
    --Crop_patches "${CROP_SIZE}" \
    --learning_rate "${LR}" \
    --print_frequency "${PRINT_FREQ}" \
    --val_frequency "${VAL_FREQ}" \
    --save_frequency "${SAVE_FREQ}" \
    --diffusion_T "${DIFFUSION_T}" \
    --lambda_reg "${LAMBDA_REG}" \
    --gen_prob "${GEN_PROB}" \
    --importance_batches "${IMPORTANCE_BATCHES}" \
    --lambda_fft "${LAMBDA_FFT}"
else
  torchrun --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --nproc_per_node="${GPUS_PER_NODE}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" train_finetune.py \
    --task "${TASK}" \
    --pre_model "${PRE_MODEL}" \
    --training_in_path "${TRAIN_IN}" \
    --training_gt_path "${TRAIN_GT}" \
    --eval_in_path "${EVAL_IN}" \
    --eval_gt_path "${EVAL_GT}" \
    --unified_path "${OUT_DIR}" \
    --writer_dir "${WRITER_DIR}" \
    --experiment_name "${EXP_NAME}" \
    --total_iters "${TOTAL_ITERS}" \
    --BATCH_SIZE "${BATCH_SIZE}" \
    --Crop_patches "${CROP_SIZE}" \
    --learning_rate "${LR}" \
    --print_frequency "${PRINT_FREQ}" \
    --val_frequency "${VAL_FREQ}" \
    --save_frequency "${SAVE_FREQ}" \
    --diffusion_T "${DIFFUSION_T}" \
    --lambda_reg "${LAMBDA_REG}" \
    --gen_prob "${GEN_PROB}" \
    --importance_batches "${IMPORTANCE_BATCHES}" \
    --lambda_fft "${LAMBDA_FFT}"
fi
