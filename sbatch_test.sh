#!/bin/bash
#SBATCH --account=jureap133
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=244
#SBATCH --time=0:10:00
#SBATCH --mem=0

srun singularity_test.sh
