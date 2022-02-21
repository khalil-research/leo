from argparse import ArgumentParser

import numpy as np
from sklearn.linear_model import LinearRegression

from const import datasets_dict
from utils import eval_learning_metrics, eval_rank_metrics
from utils import flatten_data, unflatten_data
from utils import get_unnormalized_variable_rank


def train_lr(args):
    # Load data
    dataset = datasets_dict[args.dataset]

    assert 'train' in dataset
    # array_id x # samples x # items x # features
    x_train = [np.load(_dataset['X']) for _dataset in dataset['train']]
    y_train = [np.load(_dataset['Y']) for _dataset in dataset['train']]
    y_train_shape = [_y_train.shape for _y_train in y_train]
    # 2D data: (array_id x # samples x # items) x # features
    x_train_flat, y_train_flat, weights_train_flat = flatten_data(x_train, y_train,
                                                                  loss_weights_type=args.loss_weights_type)

    x_val_flat, y_val_flat, y_val, y_val_shape = None, None, None, None
    if 'val' in dataset:
        x_val = [np.load(_dataset['X']) for _dataset in dataset['val']]
        y_val = [np.load(_dataset['Y']) for _dataset in dataset['val']]
        y_val_shape = [_y_val.shape for _y_val in y_val]
        x_val_flat, y_val_flat, weights_val_flat = flatten_data(x_val, y_val,
                                                                loss_weights_type=args.loss_weights_type)

    x_test_flat, y_test_flat, y_test, y_test_shape = None, None, None, None
    if 'test' in dataset:
        x_test = [np.load(_dataset['X']) for _dataset in dataset['test']]
        y_test = [np.load(_dataset['Y']) for _dataset in dataset['test']]
        y_test_shape = [_y_test.shape for _y_test in y_test]
        x_test_flat, y_test_flat, weights_test_flat = flatten_data(x_test, y_test,
                                                                   loss_weights_type=args.loss_weights_type)

    print('Linear Regression')
    metrics = {'train': None, 'test': None}
    reg = LinearRegression()
    reg.fit(x_train_flat, y_train_flat, sample_weight=weights_train_flat)
    print()
    print('Training performance...')
    y_train_pred = reg.predict(x_train_flat)
    metrics['train'] = eval_learning_metrics(y_train_flat, y_train_pred, weights_train_flat)

    y_val_pred = None
    if y_val_flat is not None:
        print()
        print('Val performance...')
        y_val_pred = reg.predict(x_val_flat)
        metrics['val'] = eval_learning_metrics(y_val_flat, y_val_pred, weights_val_flat)

    y_test_pred = None
    if x_test_flat is not None:
        print()
        print('Test performance...')
        y_test_pred = reg.predict(x_test_flat)
        metrics['test'] = eval_learning_metrics(y_test_flat, y_test_pred, weights_test_flat)

    y_pred_unflattened = {
        'train': unflatten_data(y_train_pred, y_train_shape),
        'val': unflatten_data(y_val_pred, y_val_shape),
        'test': unflatten_data(y_test_pred, y_test_shape)}
    y_pred_unflattened_unnorm_rank = get_unnormalized_variable_rank(y_pred_unflattened)
    y_unnorm_rank = get_unnormalized_variable_rank({'train': y_train, 'val': y_val,
                                                    'test': y_test})
    eval_rank_metrics(y_unnorm_rank, y_pred_unflattened_unnorm_rank)

    # pkl.dump(y_pred_unflattened, open(f'predictions/LR_norm_rank_{args.dataset}.pkl', 'wb'))
    # pkl.dump(y_pred_unflattened_unnorm_rank, open(f'predictions/LR_unnorm_rank_{args.dataset}.pkl', 'wb'))
    # pkl.dump(reg, open(f'pretrained/LR_{args.dataset}.pkl', 'wb'))
    # pkl.dump(metrics, open(f'metrics/LR_{args.dataset}.pkl', 'wb'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--loss_weights_type', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='3_60')
    args = parser.parse_args()

    train_lr(args)
