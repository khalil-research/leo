import logging

import numpy as np
import pandas as pd
# import scipy as sp
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

log = logging.getLogger(__name__)


def eval_learning_metrics(orig, pred, sample_weight):
    mse = mean_squared_error(orig, pred)
    mae = mean_absolute_error(orig, pred)
    mape = mean_absolute_percentage_error(orig, pred)
    r2 = r2_score(orig, pred, sample_weight=sample_weight)

    log.info(f'MSE: {mse}')
    log.info(f'MAE: {mae}')
    log.info(f'MAPE: {mape}')
    log.info(f'R2: {r2}')

    return {'mse': mse, 'r2': r2, 'mae': mae, 'mape': mape}


def count_same(x, y, start=0, end=10):
    """Count the number of indices which correct variable prediction"""
    assert len(x.shape) == 1 and len(y.shape) == 1

    return np.sum(x[start:end] == y[start:end])


def count_common(x, y, start=0, end=10):
    """Count the number of variables common in x and y from [start, end)"""
    assert len(x.shape) == 1 and len(y.shape) == 1

    return len(set(x[start:end]).intersection(set(y[start:end])))


def eval_penalty(orig, pred, start=0, end=10):
    """Evaluate how far the variable is located in the predicted order as
    compared to its position in the original order from [start, end)"""
    assert len(orig.shape) == 1 and len(pred.shape) == 1

    penalties = []
    for i in range(start, end):
        penalties.append(np.abs(i - np.where(pred == orig[i])[0][0]))

    return np.mean(penalties)


def eval_order_metrics(y_orders, y_orders_pred, n_items):
    metrics = []
    for idx, (y_order, y_order_pred) in enumerate(zip(y_orders, y_orders_pred)):
        # Spearman rank correlation
        # corr, p = sp.stats.spearmanr(y, y_hat)

        # Top 10 accuracy
        y_order, y_order_pred = y_order[:n_items[idx]], y_order_pred[:n_items[idx]]
        top_10_common = count_common(y_order, y_order_pred, end=10)
        top_10_same = count_same(y_order, y_order_pred, end=10)
        top_10_penalty = eval_penalty(y_order, y_order_pred, end=10)

        # Top 5 accuracy
        top_5_common = count_common(y_order, y_order_pred, end=5)
        top_5_same = count_same(y_order, y_order_pred, end=5)
        top_5_penalty = eval_penalty(y_order, y_order_pred, end=5)

        # metrics.append([idx, 'corr', corr])
        # metrics.append([idx, 'p', corr])
        metrics.append([idx, 'top_10_common', top_10_common])
        metrics.append([idx, 'top_10_same', top_10_same])
        metrics.append([idx, 'top_10_penalty', top_10_penalty])
        metrics.append([idx, 'top_5_common', top_5_common])
        metrics.append([idx, 'top_5_same', top_5_same])
        metrics.append([idx, 'top_5_penalty', top_5_penalty])

    df = pd.DataFrame(metrics, columns=['id', 'metric_type', 'metric_value'])
    # log.info(f"Correlation    : {df[df['metric_type'] == 'corr']['metric_value'].mean()} +/- "
    #          f"{df[df['metric_type'] == 'corr']['metric_value'].std()}")
    # log.info(f"p-value        : {df[df['metric_type'] == 'p']['metric_value'].mean()} +/- "
    #          f"{df[df['metric_type'] == 'p']['metric_value'].std()}")
    log.info(f"Top 10 Common  : {df[df['metric_type'] == 'top_10_common']['metric_value'].mean()} +/- "
             f"{df[df['metric_type'] == 'top_10_common']['metric_value'].std()} ")
    log.info(
        f"Top 10 Same : {df[df['metric_type'] == 'top_10_same']['metric_value'].mean()} +/- "
        f"{df[df['metric_type'] == 'top_10_same']['metric_value'].std()} ")
    log.info(
        f"Top 10 Penalty    : {df[df['metric_type'] == 'top_10_penalty']['metric_value'].mean()} +/- "
        f"{df[df['metric_type'] == 'top_10_penalty']['metric_value'].std()} ")
    log.info(
        f"Top 5 Common   : {df[df['metric_type'] == 'top_5_common']['metric_value'].mean()} +/- "
        f"{df[df['metric_type'] == 'top_5_common']['metric_value'].std()} ")
    log.info(
        f"Top 5 Same  : {df[df['metric_type'] == 'top_5_same']['metric_value'].mean()} +/- "
        f"{df[df['metric_type'] == 'top_5_same']['metric_value'].std()} ")
    log.info(
        f"Top 5 Penalty     : {df[df['metric_type'] == 'top_5_penalty']['metric_value'].mean()} +/- "
        f"{df[df['metric_type'] == 'top_5_penalty']['metric_value'].std()} ")

    return metrics
