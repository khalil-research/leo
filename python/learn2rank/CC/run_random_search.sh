#!/bin/bash
#SBATCH --account=rrg-khalile2
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:p100:1

# Load modules
module load python/3.8
module load meta-farm

# Activate env
source ~/envs/l2o/bin/activate

# For meta
task.run