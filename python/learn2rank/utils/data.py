from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

ROOT_PATH = Path(__file__).parent.parent


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


def read_data_from_file(filepath):
    data = {'value': [], 'weight': [], 'capacity': 0}

    with open(filepath, 'r') as fp:
        n_vars = int(fp.readline())
        n_objs = int(fp.readline())
        for _ in range(n_objs):
            data['value'].append(list(map(int, fp.readline().split())))
        data['weight'].extend(list(map(int, fp.readline().split())))
        data['capacity'] = int(fp.readline().split()[0])

    return data


def flatten_data(X, Y, weighted_loss=0):
    X_flat, Y_flat, weights_flat = [], [], []
    for x, y in zip(X, Y):
        scale_y = np.max(y) - np.min(y)
        for item_x, item_y in zip(x, y):
            weight = 1
            if weighted_loss == 1:
                """Linearly decreasing"""
                weight = 1 - (item_y / scale_y)
            elif weighted_loss == 2:
                """Exponentially decreasing"""
                weight = np.exp(-((item_y / scale_y) * 5))
            weights_flat.append(weight)

            X_flat.append(item_x)
            Y_flat.append(item_y)

    X_flat, Y_flat = np.asarray(X_flat), np.asarray(Y_flat)
    weights_flat = np.asarray(weights_flat)
    print(X_flat.shape, Y_flat.shape)

    return X_flat, Y_flat, weights_flat


def unflatten_data(Y, num_items):
    Y_out = []
    num_samples = int(Y.shape[0] / num_items)
    if Y is not None:
        i = 0
        for j in range(num_samples):
            Y_out.append(Y[i: i + num_items])
            i += num_items

    return Y_out


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
