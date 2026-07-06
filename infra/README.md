# DDL Distributed Training Infra

This folder provides ready-to-use launch templates for single-node multi-GPU,
multi-node multi-GPU, and SLURM training.

## Requirements

- Linux training environment with CUDA, NCCL, PyTorch distributed, and `torchrun`.
- The same DDL code path on every node.
- Dataset and checkpoint paths visible from every node, preferably through shared storage.
- `MASTER_ADDR` reachable from every worker node.
- The selected `MASTER_PORT` open between nodes.

The training code already uses `DistributedDataParallel` and
`DistributedSampler`. The scripts in this folder configure the missing cluster
layer around the existing DDL training entry points.

## Files

| File | Purpose |
|---|---|
| `distributed.env` | Common NCCL and runtime environment variables |
| `launch_pretrain.sh` | `torchrun` launcher for denoising pre-training |
| `launch_finetune.sh` | `torchrun` launcher for single-task GEF fine-tuning |
| `launch_unified.sh` | `torchrun` launcher for multi-task unified training |
| `slurm_pretrain.sbatch` | SLURM job template for pre-training |
| `slurm_finetune.sbatch` | SLURM job template for fine-tuning |
| `slurm_unified.sbatch` | SLURM job template for unified training |
| `hostfile.example` | Example host list for documenting multi-node resources |

## Single-node 8-GPU

Pre-training:

```bash
GPUS_PER_NODE=8 \
GT_DIR=/path/to/data/gt_images \
SAVE_PATH=./ckpt \
TOTAL_ITERS=100000 \
BATCH_SIZE=16 \
bash infra/launch_pretrain.sh
```

Single-task fine-tuning:

```bash
GPUS_PER_NODE=8 \
TASK=dehazing \
PRE_MODEL=./ckpt/pretrained_model.pth \
TRAIN_IN=/path/to/data/dehazing/train_input \
TRAIN_GT=/path/to/data/dehazing/train_gt \
EVAL_IN=/path/to/data/dehazing/val_input \
EVAL_GT=/path/to/data/dehazing/val_gt \
EXP_NAME=ddl_gef_dehazing_8gpu \
TOTAL_ITERS=500000 \
BATCH_SIZE=16 \
bash infra/launch_finetune.sh
```

Multi-task unified training:

```bash
GPUS_PER_NODE=8 \
PRE_MODEL=./ckpt/pretrained_model.pth \
DATA_ROOT=/path/to/data \
EXP_NAME=ddl_unified_8gpu \
ITERS_PER_TASK=100000 \
BATCH_SIZE=16 \
bash infra/launch_unified.sh
```

## Multi-node torchrun

Run the same command on every node and change only `NODE_RANK`.

Node 0:

```bash
NNODES=2 \
NODE_RANK=0 \
GPUS_PER_NODE=8 \
MASTER_ADDR=10.0.0.1 \
MASTER_PORT=29500 \
DATA_ROOT=/shared/data \
PRE_MODEL=/shared/ckpt/pretrained_model.pth \
bash infra/launch_unified.sh
```

Node 1:

```bash
NNODES=2 \
NODE_RANK=1 \
GPUS_PER_NODE=8 \
MASTER_ADDR=10.0.0.1 \
MASTER_PORT=29500 \
DATA_ROOT=/shared/data \
PRE_MODEL=/shared/ckpt/pretrained_model.pth \
bash infra/launch_unified.sh
```

The same pattern works for `launch_pretrain.sh` and `launch_finetune.sh`.

## SLURM

Edit the `#SBATCH --partition`, `#SBATCH --nodes`, `#SBATCH --gres`, and time
limit fields in the corresponding `.sbatch` file. Then submit:

```bash
sbatch infra/slurm_pretrain.sbatch
sbatch infra/slurm_finetune.sbatch
sbatch infra/slurm_unified.sbatch
```

You can override training arguments when submitting:

```bash
sbatch --export=ALL,DATA_ROOT=/shared/data,PRE_MODEL=/shared/ckpt/pretrained_model.pth,EXP_NAME=ddl_unified_2n16g infra/slurm_unified.sbatch
```

## Batch Size and Learning Rate

`BATCH_SIZE` in DDL scripts is the per-process batch size. The effective global
batch size is:

```text
global_batch = BATCH_SIZE * GPUS_PER_NODE * NNODES
```

The launch scripts do not automatically scale the learning rate. If you change
the number of GPUs or per-GPU batch size substantially, set `LR` explicitly.

## Common Variables

| Variable | Default | Meaning |
|---|---:|---|
| `NNODES` | `1` | Number of nodes |
| `NODE_RANK` | `0` | Rank of the current node |
| `GPUS_PER_NODE` | `8` | GPU processes per node |
| `MASTER_ADDR` | `127.0.0.1` | Address of rank-0 node |
| `MASTER_PORT` | `29500` | Distributed rendezvous port |
| `BATCH_SIZE` | `16` | Batch size per GPU process |
| `LR` | `5e-5` | Learning rate |
| `EXP_NAME` | stage-specific | Experiment name |
| `OUT_DIR` | `./experiments` | Checkpoint root for fine-tuning/unified training |
| `WRITER_DIR` | `./tf-logs` | TensorBoard log root |

## Troubleshooting

`Address already in use`: change `MASTER_PORT`.

`NCCL timeout`: verify that all nodes can reach `MASTER_ADDR:MASTER_PORT` and
that the dataset path is visible on every node.

`CUDA out of memory`: reduce `BATCH_SIZE`, `CROP_SIZE`, or `GPUS_PER_NODE`.

`Hanging at startup`: make sure every node runs the command with the same
`NNODES`, `MASTER_ADDR`, `MASTER_PORT`, and a unique `NODE_RANK`.

