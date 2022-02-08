import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from featurizer import get_features
from utils import read_from_file
from utils_order import get_variable_rank


def main(opts):
    XX, YY = [], []
    for i in range(opts.n_instances):
        # Read data file
        filepath = f'{opts.data_path}/kp_{opts.opt_seed}_{opts.p}_{opts.n}_{i}.dat'
        data = read_from_file(opts.p, filepath)
        # Get features
        X = get_features(opts, data)
        XX.append(X)

        # Prepare y
        # Load incumbent config
        folder = f'kp_{opts.opt_seed}_{opts.p}_{opts.n}_{i}/run_{opts.smac_seed}'
        smac_out_dir = Path(opts.output_dir).joinpath(folder)
        traj = smac_out_dir.joinpath('traj.json')
        with open(traj, 'r') as fp:
            lines = fp.readlines()
        best_incumbent = json.loads(lines[-1])
        best_incumbent = best_incumbent['incumbent']
        variables_rank = get_variable_rank(data, best_incumbent)
        YY.append(variables_rank)

    XX = np.asarray(XX)
    YY = np.asarray(YY)
    np.save('X_train_3_60_r1_c2_A_ta.npy', XX)
    np.save('Y_train_3_60_r1_c2_A_ta.npy', YY)
    print(XX.shape, YY.shape)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='/home/rahul/Documents/PhD/projects/multiobjective_cp2016/data/knapsack_7/3_60/train')
    parser.add_argument('--output_dir', type=str,
                        default='/home/rahul/Documents/PhD/projects/multiobjective_cp2016/output/3_60_r1_c2_A/train')
    parser.add_argument('--n_instances', type=int, default=1)
    parser.add_argument('--p', type=int, default=3)
    parser.add_argument('--n', type=int, default=60)
    parser.add_argument('--p_max', type=int, default=7)
    parser.add_argument('--n_max', type=int, default=100)
    parser.add_argument('--opt_seed', type=int, default=7)
    parser.add_argument('--smac_seed', type=int, default=777)
    opts = parser.parse_args()
    main(opts)
