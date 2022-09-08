import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from learn2rank.utils import eval_learning_metrics
from learn2rank.utils import eval_rank_metrics
from learn2rank.utils import flatten_data
from learn2rank.utils import get_unnormalized_variable_rank
from learn2rank.utils import numpy_dataset_paths
from learn2rank.utils import unflatten_data


def train_lr(args):
    # Load data
    num_items = int(args.dataset_name.split("_")[1])
    dataset = numpy_dataset_paths[args.dataset_name]
    root_path = Path(__file__).parent.parent
    assert 'train' in dataset
    # # samples x # items x # features
    x_train = np.load(str(root_path.joinpath(dataset['train']['X'])))
    y_train = np.load(str(root_path.joinpath(dataset['train']['Y'])))
    # 2D data: (# samples x # items) x # features
    x_train, y_train, weights_train = flatten_data(x_train, y_train,
                                                   weighted_loss=args.weighted_loss)

    x_val = np.load(str(root_path.joinpath(dataset['val']['X'])))
    y_val = np.load(str(root_path.joinpath(dataset['val']['Y'])))
    x_val, y_val, weights_val = flatten_data(x_val, y_val,
                                             weighted_loss=args.weighted_loss)

    x_test = np.load(str(root_path.joinpath(dataset['test']['X'])))
    y_test = np.load(str(root_path.joinpath(dataset['test']['Y'])))
    x_test, y_test, weights_test = flatten_data(x_test, y_test,
                                                weighted_loss=args.weighted_loss)

    print('Linear Regression')
    metrics = {'train': None, 'val': None, 'test': None}
    reg = LinearRegression()
    reg.fit(x_train, y_train, sample_weight=weights_train)

    print()
    print('Training performance...')
    y_train_pred = reg.predict(x_train)
    metrics['train'] = eval_learning_metrics(y_train, y_train_pred, weights_train)

    print()
    print('Val performance...')
    y_val_pred = reg.predict(x_val)
    metrics['val'] = eval_learning_metrics(y_val, y_val_pred, weights_val)

    print()
    print('Test performance...')
    y_test_pred = reg.predict(x_test)
    metrics['test'] = eval_learning_metrics(y_test, y_test_pred, weights_test)

    y_pred_unflattened = {
        'train': unflatten_data(y_train_pred, num_items),
        'val': unflatten_data(y_val_pred, num_items),
        'test': unflatten_data(y_test_pred, num_items)
    }
    y_pred_unnorm_rank = {
        'train': get_unnormalized_variable_rank(y_pred_unflattened['train']),
        'val': get_unnormalized_variable_rank(y_pred_unflattened['val']),
        'test': get_unnormalized_variable_rank(y_pred_unflattened['test'])
    }

    y_unflattened = {
        'train': unflatten_data(y_train, num_items),
        'val': unflatten_data(y_val, num_items),
        'test': unflatten_data(y_test, num_items)
    }
    y_unnorm_rank = {
        'train': get_unnormalized_variable_rank(y_unflattened['train']),
        'val': get_unnormalized_variable_rank(y_unflattened['val']),
        'test': get_unnormalized_variable_rank(y_unflattened['test'])
    }

    print()
    print(f'Train rank metrics...')
    metrics.update(eval_rank_metrics(y_unnorm_rank['train'], y_pred_unnorm_rank['train']))

    print()
    print(f'Validation rank metrics...')
    metrics.update(eval_rank_metrics(y_unnorm_rank['val'], y_pred_unnorm_rank['val']))

    print()
    print(f'Test rank metrics...')
    metrics.update(eval_rank_metrics(y_unnorm_rank['test'], y_pred_unnorm_rank['test']))

    pkl.dump(y_pred_unflattened,
             open(f'{str(root_path)}/resources/predictions/LR_norm_rank_{args.dataset_name}.pkl', 'wb'))
    pkl.dump(y_pred_unnorm_rank,
             open(f'{str(root_path)}/resources/predictions/LR_unnorm_rank_{args.dataset_name}.pkl', 'wb'))
    pkl.dump(reg,
             open(f'{str(root_path)}/resources/pretrained/LR_{args.dataset_name}.pkl', 'wb'))
    pkl.dump(metrics,
             open(f'{str(root_path)}/resources/metrics/LR_{args.dataset_name}.pkl', 'wb'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weighted_loss', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='3_60_mat')
    args = parser.parse_args()

    train_lr(args)
