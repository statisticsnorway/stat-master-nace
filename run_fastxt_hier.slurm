#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --account=ec403
#SBATCH --time=3:30:00
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
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

module load nlpl-nlptools/01-foss-2024a-Python-3.12.3 #Empty or non-existent file

module load nlpl-scipy-ecosystem/01-foss-2024a-Python-3.12.3
module load nlpl-llmtools/01-foss-2024a-Python-3.12.3
module load nlpl-sentence-transformers/5.1.1-foss-2024a-Python-3.12.3
module load nlpl-tokenizers/0.21.4-foss-2024a-Python-3.12.3



# print information (optional)
echo "submission directory: ${SUBMITDIR}"

# by default, pass on any remaining command-line options
python3 run_fasttxt_hier.py ${@}