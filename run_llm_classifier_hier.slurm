#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --account=ec403
#SBATCH --time 10:30:00
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=4

# NB: this script should be run with "sbatch run_classifier.slurm"!
# See https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/jobs/submitting.md

source ~/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/

module load nlpl-nlptools/01-foss-2024a-Python-3.12.3 #Empty or non-existent file

module load nlpl-scipy-ecosystem/01-foss-2024a-Python-3.12.3
module load nlpl-llmtools/01-foss-2024a-Python-3.12.3
module load nlpl-sentence-transformers/5.1.1-foss-2024a-Python-3.12.3
module load nlpl-tokenizers/0.21.4-foss-2024a-Python-3.12.3
module load nlpl-vllm/0.8.2-foss-2024a-Python-3.12.3


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

export VLLM_WORKER_MULTIPROC_METHOD=spawn


srun --label python3 llm_classifier_hier.py $@