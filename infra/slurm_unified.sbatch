#!/usr/bin/env bash
#SBATCH --job-name=ddl_unified
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${SLURM_NNODES}
export NODE_RANK=${SLURM_NODEID}
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}

bash infra/launch_unified.sh
