#!/bin/bash
#SBATCH --account=def-khalile2
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0-49

module load python/3.8
source ~/envs/l2o/bin/activate

python eval_predicted_orderings.py --raw_data /scratch/rahulpat/knapsack_7 --split test --from_pid 1100 --num_instances 100 --predictions 'predictions/LR_3_60_unnorm_rank.pkl' --time_limit 600