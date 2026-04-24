#!/bin/bash
#SBATCH --job-name=nace_classifier_hier
#SBATCH --account=ec403
#SBATCH --time 20:30:00
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100_80:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=gpu-14,gpu-5,gpu-11,gpu-7,gpu-17

# NB: this script should be run with "sbatch run_classifier_hier.slurm"!
# See https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/jobs/submitting.md

source ~/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/

module load nlpl-nlptools/01-foss-2024a-Python-3.12.3 

module load nlpl-scipy-ecosystem/01-foss-2024a-Python-3.12.3
module load nlpl-llmtools/01-foss-2024a-Python-3.12.3
module load nlpl-sentence-transformers/5.1.1-foss-2024a-Python-3.12.3
module load nlpl-tokenizers/0.21.4-foss-2024a-Python-3.12.3
module load nlpl-vllm/0.8.2-foss-2024a-Python-3.12.3

nvidia-smi


export CACHE_ROOT=/cluster/work/projects/ec403/ec-minashe-master
export TRITON_CACHE_DIR=$CACHE_ROOT/triton_cache
export HF_HOME=$CACHE_ROOT/hf_cache
export VLLM_CACHE_ROOT=$CACHE_ROOT/vllm_cache

export HF_TOKEN="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" 

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

export LC_ALL=en_US.UTF-8

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_CUDAGRAPH=1 # added
export CUDA_LAUNCH_BLOCKING=1

srun --label python3 llm_classifier_hier.py $@