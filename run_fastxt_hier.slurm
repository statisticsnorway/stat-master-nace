#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --account=ec403
#SBATCH --time=20:30:00
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
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

module load nlpl-nlptools/04-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
module load nlpl-gensim/4.3.2-foss-2022b-Python-3.10.8
module load nlpl-transformers/4.43.4-foss-2022b-Python-3.10.8 
module load nlpl-accelerate/0.33.0-foss-2022b-Python-3.10.8 
module load nlpl-torchmetrics/1.2.1-foss-2022b-Python-3.10.8


# print information (optional)
echo "submission directory: ${SUBMITDIR}"

# by default, pass on any remaining command-line options
python3 run_fasttxt_hier.py ${@}