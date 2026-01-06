#!/bin/bash
#SBATCH --job-name=test_vllm
#SBATCH --account=project_465002259
#SBATCH --partition=small-g
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --gpus-per-node=4


module purge
export EBU_USER_PREFIX=/projappl/project_465002259/software/
module load LUMI
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404

export TRITON_CACHE_DIR="./triton_cache"
export HF_HOME="./hf_cache"
export VLLM_CACHE_ROOT="./vllm_cache"

export HF_TOKEN="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" 

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

export LC_ALL=en_US.UTF-8

srun --label python3 scorer.py $@