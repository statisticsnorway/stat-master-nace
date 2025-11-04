#!/bin/bash
#SBATCH --job-name=test_vllm
#SBATCH --account=project_465002259
#SBATCH --partition=dev-g
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1


export EBU_USER_PREFIX=/project/project_465001925/charpent2.6
module purge
module load LUMI
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-singularity-20250404
module load SciPy-bundle/2025.06-gfbf-2025a

module load nlpl-nlptools/04-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
module load nlpl-transformers/4.43.4-foss-2022b-Python-3.10.8 
module load nlpl-accelerate/0.33.0-foss-2022b-Python-3.10.8 
module load nlpl-torchmetrics/1.2.1-foss-2022b-Python-3.10.8
module load nlpl-llmtools/06-foss-2022b-Python-3.10.8 
module load nlpl-datasets/3.2.0-foss-2022b-Python-3.10.8 
module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8 
module load nlpl-trl/0.15.2-foss-2022b-Python-3.10.8
module load nlpl-torchmetrics/1.2.1-foss-2022b-Python-3.10.8


export TRITON_CACHE_DIR="./triton_cache"
export HF_HOME="./hf_cache"
export VLLM_CACHE_ROOT="./vllm_cache"

export HF_TOKEN="hf_XXX"  # Replace XXX with your actual Hugging Face token

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

export LC_ALL=en_US.UTF-8


srun --label python3 run_fasttxt_hier.py $@