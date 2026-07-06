#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source infra/distributed.env

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

PRE_MODEL=${PRE_MODEL:-./ckpt/pretrained_model.pth}
DATA_ROOT=${DATA_ROOT:-./data}
TASKS=${TASKS:-noisy,rainy,jpeg,snowy,inpainting,raindrop,shadowed,lowlight,hazy,blurry}
OUT_DIR=${OUT_DIR:-./experiments}
WRITER_DIR=${WRITER_DIR:-./tf-logs}
EXP_NAME=${EXP_NAME:-ddl_unified_ddp}
ITERS_PER_TASK=${ITERS_PER_TASK:-100000}
BATCH_SIZE=${BATCH_SIZE:-16}
CROP_SIZE=${CROP_SIZE:-128}
LR=${LR:-5e-5}
PRINT_FREQ=${PRINT_FREQ:-200}
VAL_FREQ=${VAL_FREQ:-5000}
NUM_EXPERTS=${NUM_EXPERTS:-10}
DIFFUSION_T=${DIFFUSION_T:-50}
LAMBDA_REG=${LAMBDA_REG:-0.2}
GEN_PROB=${GEN_PROB:-0.1}
LAMBDA_FFT=${LAMBDA_FFT:-0.1}

if [ "${NNODES}" = "1" ]; then
  torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" train_unified.py \
    --pre_model "${PRE_MODEL}" \
    --data_root "${DATA_ROOT}" \
    --tasks "${TASKS}" \
    --unified_path "${OUT_DIR}" \
    --writer_dir "${WRITER_DIR}" \
    --experiment_name "${EXP_NAME}" \
    --iters_per_task "${ITERS_PER_TASK}" \
    --BATCH_SIZE "${BATCH_SIZE}" \
    --Crop_patches "${CROP_SIZE}" \
    --learning_rate "${LR}" \
    --print_frequency "${PRINT_FREQ}" \
    --val_frequency "${VAL_FREQ}" \
    --num_experts "${NUM_EXPERTS}" \
    --diffusion_T "${DIFFUSION_T}" \
    --lambda_reg "${LAMBDA_REG}" \
    --gen_prob "${GEN_PROB}" \
    --lambda_fft "${LAMBDA_FFT}"
else
  torchrun --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --nproc_per_node="${GPUS_PER_NODE}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" train_unified.py \
    --pre_model "${PRE_MODEL}" \
    --data_root "${DATA_ROOT}" \
    --tasks "${TASKS}" \
    --unified_path "${OUT_DIR}" \
    --writer_dir "${WRITER_DIR}" \
    --experiment_name "${EXP_NAME}" \
    --iters_per_task "${ITERS_PER_TASK}" \
    --BATCH_SIZE "${BATCH_SIZE}" \
    --Crop_patches "${CROP_SIZE}" \
    --learning_rate "${LR}" \
    --print_frequency "${PRINT_FREQ}" \
    --val_frequency "${VAL_FREQ}" \
    --num_experts "${NUM_EXPERTS}" \
    --diffusion_T "${DIFFUSION_T}" \
    --lambda_reg "${LAMBDA_REG}" \
    --gen_prob "${GEN_PROB}" \
    --lambda_fft "${LAMBDA_FFT}"
fi
