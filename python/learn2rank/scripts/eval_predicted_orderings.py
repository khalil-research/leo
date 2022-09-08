import os
import pickle as pkl
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from utils.const import datasets_dict
from utils import get_order_from_rank
from utils import run_bdd_builder


def eval_predictions(args):
    random_seeds = [13, 444, 1212, 1003, 7517]
    dataset = datasets_dict[args.dataset]
    predictions = pkl.load(open(args.predictions, 'rb'))

    for _dataset, _predictions in zip(dataset[args.split], predictions[args.split]):
        data_path = Path(args.raw_data) / _dataset['id'] / args.split
        assert data_path.exists()

        # Total number of slurm workers detected
        # Defaults to 1 if not running under SLURM
        N_WORKERS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

        # This worker's array index. Assumes slurm array job is zero-indexed
        # Defaults to zero if not running under SLURM
        this_worker = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))

        for idx, i in enumerate(range(this_worker + args.from_pid,
                                      args.from_pid + args.num_instances, N_WORKERS)):
            instance_path = data_path / f'kp_7_{_dataset["id"]}_{i}.dat'

            order = get_order_from_rank(_predictions[idx])
            status, runtime = run_bdd_builder(instance_path, order,
                                              time_limit=args.time_limit,
                                              mem_limit=args.mem_limit)
            print(str(instance_path), 'BEST INC', status, runtime)

            for ridx, rseed in enumerate(random_seeds):
                random_order = list(
                    np.arange(int(_dataset['id'].split('_')[1])))
                random.seed(rseed)
                random.shuffle(random_order)
                status, runtime = run_bdd_builder(instance_path, random_order,
                                                  time_limit=args.time_limit,
                                                  mem_limit=args.mem_limit)
                print(str(instance_path), f"rnd{ridx}", status, runtime)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='3_60')
    parser.add_argument('--raw_data', type=str,
                        default='/home/rahul/Documents/PhD/projects/multiobjective_cp2016/data/knapsack_7/')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--from_pid', type=int, default=1100)
    parser.add_argument('--num_instances', type=int, default=100)
    parser.add_argument('--predictions', type=str,
                        default='predictions/LR_3_60_unnorm_rank.pkl')
    # parser.add_argument('--num_objectives', type=int, default=3)
    # parser.add_argument('--num_items', type=int, default=20)
    # parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--time_limit', type=int, default=60,
                        help='Time limit in seconds')
    parser.add_argument('--mem_limit', type=int, default=16,
                        help='Memory limit in GB')
    args = parser.parse_args()

    eval_predictions(args)
