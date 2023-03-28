#!/bin/bash
#SBATCH --account=def-khalile2
#SBATCH --time=7:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --array=0-99

module load python/3.8
source ~/envs/l2o/bin/activate

python eval_static_orderings.py --time_limit 600 --mem_limit 16 --dataset /scratch/rahulpat/knapsack_7 --num_objectives 3 --num_items 20
python eval_static_orderings.py --time_limit 600 --mem_limit 16 --dataset /scratch/rahulpat/knapsack_7 --num_objectives 5 --num_items 20
python eval_static_orderings.py --time_limit 600 --mem_limit 16 --dataset /scratch/rahulpat/knapsack_7 --num_objectives 7 --num_items 20


python eval_static_orderings.py --time_limit 600 --mem_limit 16 --dataset /scratch/rahulpat/knapsack_7 --num_objectives 3 --num_items 40
python eval_static_orderings.py --time_limit 600 --mem_limit 16 --dataset /scratch/rahulpat/knapsack_7 --num_objectives 3 --num_items 60
python eval_static_orderings.py --time_limit 600 --mem_limit 16 --dataset /scratch/rahulpat/knapsack_7 --num_objectives 3 --num_items 80


#python eval_static_orderings.py --time_limit 600 --mem_limit 16 --dataset /scratch/rahulpat/knapsack_7 --num_objectives 5 --num_items 30
#python eval_static_orderings.py --time_limit 600 --mem_limit 16 --dataset /scratch/rahulpat/knapsack_7 --num_objectives 7 --num_items 30


#python eval_static_orderings.py --time_limit 600 --mem_limit 16 --dataset /scratch/rahulpat/knapsack_7 --num_objectives 5 --num_items 40
#python eval_static_orderings.py --time_limit 600 --mem_limit 16 --dataset /scratch/rahulpat/knapsack_7 --num_objectives 7 --num_items 40

