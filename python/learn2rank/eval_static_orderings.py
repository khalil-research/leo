import os
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from utils import get_static_orders
from utils import read_from_file
from utils import run_bdd_builder


def eval_static(args):
    random_seeds = [13, 444, 1212, 1003, 7517]
    data_path = Path(args.dataset) / f"{args.num_objectives}_{args.num_items}" / args.split
    assert data_path.exists()

    # Total number of slurm workers detected
    # Defaults to 1 if not running under SLURM
    N_WORKERS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

    # This worker's array index. Assumes slurm array job is zero-indexed
    # Defaults to zero if not running under SLURM
    this_worker = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))

    for i in range(this_worker, args.num_eval_instances, N_WORKERS):
        instance_path = data_path / f"kp_7_{args.num_objectives}_{args.num_items}_{i}.dat"

        data = read_from_file(args.num_objectives, instance_path)
        static_orders = get_static_orders(data)
        for ordering in static_orders.keys():
            order = static_orders[ordering]

            status, runtime = run_bdd_builder(instance_path, order,
                                              time_limit=args.time_limit,
                                              mem_limit=args.mem_limit)
            print(str(instance_path), ordering, status, runtime)

        for ridx, rseed in enumerate(random_seeds):
            random_order = list(np.arange(args.num_items))
            random.seed(rseed)
            random.shuffle(random_order)
            status, runtime = run_bdd_builder(instance_path, random_order,
                                              time_limit=args.time_limit,
                                              mem_limit=args.mem_limit)
            print(str(instance_path), f"rnd{ridx}", status, runtime)
        

if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='/scratch/rahulpat/knapsack_7')
    parser.add_argument('--dataset', type=str,
                        default='/home/rahul/Documents/PhD/projects/multiobjective_cp2016/data/knapsack_7/')
    parser.add_argument('--num_eval_instances', type=int, default=250)
    parser.add_argument('--num_objectives', type=int, default=3)
    parser.add_argument('--num_items', type=int, default=20)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--time_limit', type=int, default=60,
                        help='Time limit in seconds')
    parser.add_argument('--mem_limit', type=int, default=16,
                        help='Memory limit in GB')
    args = parser.parse_args()

    eval_static(args)
