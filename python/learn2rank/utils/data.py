from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

ROOT_PATH = Path(__file__).parent.parent


def read_data_from_file(problem_acronym, file_path):
    data = {'value': [], 'n_vars': 0, 'n_cons': 1, 'n_objs': 3}
    raw_data = open(file_path, 'r')

    def parse_knapsack():
        data['weight'], data['capacity'] = [], 0

        data['n_vars'] = int(raw_data.readline())
        data['n_objs'] = int(raw_data.readline())
        for _ in range(data['n_objs']):
            data['value'].append(list(map(int, raw_data.readline().split())))
        data['weight'].extend(list(map(int, raw_data.readline().split())))
        data['capacity'] = int(raw_data.readline().split()[0])

    def parse_binproblem():
        data['cons'], data['cons_mat'] = [], []

        data['n_vars'], data['n_cons'] = list(map(int, raw_data.readline().strip().split()))
        data['n_objs'] = int(raw_data.readline())
        for _ in range(data['n_objs']):
            data['value'].append(list(map(int, raw_data.readline().split())))
        for _ in range(data['n_cons']):
            n_vars_per_con = raw_data.readline()

            non_zero_vars_lst = list(map(int, raw_data.readline().strip().split()))
            data['cons'].append(non_zero_vars_lst)
            non_zero_vars_lst = np.array(non_zero_vars_lst)

            non_zero_vars_lst -= 1
            cons_mat = np.zeros(data['n_vars'])
            cons_mat[non_zero_vars_lst] = 1

            data['cons_mat'].append(cons_mat)
        data['cons_mat'] = np.asarray(data['cons_mat'])
        data['weight'] = np.sum(data['cons_mat'], axis=1)

    if problem_acronym == 'kp':
        parse_knapsack()

    elif problem_acronym == 'bp':
        parse_binproblem()

    else:
        raise ValueError('Invalid problem!')

    return data


class PointwiseVariableRankRegressionDataset(Dataset):
    def __init__(self, x, y, wt, device):
        self.x = torch.from_numpy(x).float().to(device)
        self.y = torch.from_numpy(y).float().to(device)
        self.wt = torch.from_numpy(wt).float().to(device)
        self.device = device

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.wt[idx]


class WeightRegressionDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, item):
        return 1


def normalize_labels(Y, n_max_vars, padded_value=-1):
    assert len(Y.shape) == 2

    pads = Y == padded_value
    divs = np.ones_like(Y) / n_max_vars
    divs = divs * (~pads)

    Y = Y * divs

    return Y


def flatten_data(nested_lists):
    flattened_lists = []
    for nested_list in nested_lists:
        _flattened_list = []
        for item in nested_list:
            _flattened_list.extend(item)

        _flattened_list = np.asarray(_flattened_list)
        flattened_lists.append(_flattened_list)

    return flattened_lists


def get_n_items(y, padded_value=-1):
    if type(y) == list:
        y = np.asarray(y)

    assert len(y.shape) <= 2
    if len(y.shape) == 1:
        y = y.reshape(1, -1)

    # True where -1
    mask = y == padded_value
    # False where -1
    mask = ~mask

    n_items = np.sum(mask, axis=1)
    assert n_items.shape[0] == y.shape[0]

    return n_items


def get_sample_weight(y, weighted_loss, padded_value=-1):
    if type(y) == list:
        y = np.asarray(y)

    assert len(y.shape) <= 2
    if len(y.shape) == 1:
        y = y.reshape(1, -1)

    weights = []
    for i in range(y.shape[0]):
        _weights = []

        y_min = np.min(y[i])
        y_min = y_min if y_min != padded_value else 0
        scale_y = np.max(y[i]) - y_min

        for _y in y[i]:
            wt = 1
            if _y != padded_value:
                if weighted_loss == 1:
                    """Linearly decreasing"""
                    wt = 1 - (_y / scale_y)
                elif weighted_loss == 2:
                    """Exponentially decreasing"""
                    wt = np.exp(-((_y / scale_y) * 5))
            else:
                wt = 0

            _weights.append(wt)

        weights.append(_weights)

    return np.asarray(weights)


def unflatten_data(flattened_lists, n_max_vars):
    unflattened_list = []
    for flattened_list in flattened_lists:
        _unflattened_list = []

        num_samples = int(flattened_list.shape[0] / n_max_vars)
        i = 0
        for _ in range(num_samples):
            _unflattened_list.append(flattened_list[i: i + n_max_vars])
            i += n_max_vars

        _unflattened_list = np.asarray(_unflattened_list)
        unflattened_list.append(_unflattened_list)

    return unflattened_list


# def get_dataloaders(args, tr_dataset, val_dataset, test_dataset):
#     tr_loader = DataLoader(tr_dataset, shuffle=True, batch_size=args.bs)
#     val_loader = DataLoader(val_dataset, shuffle=False, batch_size=len(val_dataset))
#     test_loader = DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset))
#
#     return tr_loader, val_loader, test_loader


def get_flattened_split(args, dataset):
    x_train = np.load(str(ROOT_PATH.joinpath(dataset['X'])))
    y_train = np.load(str(ROOT_PATH.joinpath(dataset['Y'])))
    # 2D data: (# samples x # items) x # features
    x_train, y_train, weights_train = flatten_data(x_train, y_train,
                                                   weighted_loss=args.weighted_loss)

    return x_train, y_train, weights_train


def get_dummy_data(cfg):
    return {
        'x_tr': np.random.rand(100, cfg.n_max_vars, cfg.n_features),
        'x_val': np.random.rand(100, cfg.n_max_vars, cfg.n_features),
        'x_test': np.random.rand(100, cfg.n_max_vars, cfg.n_features),
        'y_tr': {
            'weight': np.random.rand(100, cfg.n_weights),
            'time': np.random.rand(100, 1),
            'rank': np.random.rand(100, cfg.n_max_vars)
        },
        'y_val': {
            'weight': np.random.rand(100, cfg.n_weights),
            'time': np.random.rand(100, 1),
            'rank': np.random.rand(100, cfg.n_max_vars)
        },
        'y_test': {
            'weight': np.random.rand(100, cfg.n_weights),
            'time': np.random.rand(100, 1),
            'rank': np.random.rand(100, cfg.n_max_vars)
        }
    }
