from operator import itemgetter

import numpy as np

from .const import StaticOrderings
from .utils import get_rank


def get_instance_features(opts, norm_weight, norm_value):
    p, n = norm_value.shape

    features = [p / opts.p_max, n / opts.n_max,
                (np.ceil(norm_weight.sum()) / 2) / n,  # Normalized capacity
                norm_weight.mean(), norm_weight.min(), norm_weight.max(), norm_weight.std()]  # Weight aggregate stats

    # Value double-aggregate stats    
    value_mean = norm_value.mean(axis=1)
    value_min = norm_value.min(axis=1)
    value_max = norm_value.max(axis=1)

    features.extend([value_mean.mean(), value_mean.min(), value_mean.max(), value_mean.std(),
                     value_min.mean(), value_min.min(), value_min.max(), value_min.std(),
                     value_max.mean(), value_max.min(), value_max.max(), value_max.std()])

    features = np.asarray(features)

    return features


def get_item_features(data, norm_weight, norm_value):
    item_features = np.vstack([norm_weight,
                               norm_value.mean(axis=0),
                               norm_value.min(axis=0),
                               norm_value.max(axis=0),
                               norm_value.std(axis=0),
                               norm_value.mean(axis=0) / norm_weight,
                               norm_value.max(axis=0) / norm_weight,
                               norm_value.min(axis=0) / norm_weight])
    n_items = len(data['weight'])

    for o in StaticOrderings:
        if o.name == 'max_weight':
            idx_weight = [(i, w) for i, w in enumerate(data['weight'])]
            idx_weight.sort(key=itemgetter(1), reverse=True)
            idx_rank = get_rank(idx_weight)

        elif o.name == 'min_weight':
            idx_weight = [(i, w) for i, w in enumerate(data['weight'])]
            idx_weight.sort(key=itemgetter(1))
            idx_rank = get_rank(idx_weight)

        elif o.name == 'max_avg_value':
            mean_profit = np.mean(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(mean_profit)]
            idx_profit.sort(key=itemgetter(1), reverse=True)
            idx_rank = get_rank(idx_profit)

        elif o.name == 'min_avg_value':
            mean_profit = np.mean(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(mean_profit)]
            idx_profit.sort(key=itemgetter(1))
            idx_rank = get_rank(idx_profit)

        elif o.name == 'max_max_value':
            max_profit = np.max(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(max_profit)]
            idx_profit.sort(key=itemgetter(1), reverse=True)
            idx_rank = get_rank(idx_profit)

        elif o.name == 'min_max_value':
            max_profit = np.max(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(max_profit)]
            idx_profit.sort(key=itemgetter(1))
            idx_rank = get_rank(idx_profit)

        elif o.name == 'max_min_value':
            min_profit = np.min(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(min_profit)]
            idx_profit.sort(key=itemgetter(1), reverse=True)
            idx_rank = get_rank(idx_profit)

        elif o.name == 'min_min_value':
            min_profit = np.min(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(min_profit)]
            idx_profit.sort(key=itemgetter(1))
            idx_rank = get_rank(idx_profit)

        elif o.name == 'max_avg_value_by_weight':
            mean_profit = np.mean(data['value'], 0)
            profit_by_weight = [v / w for v, w in zip(mean_profit, data['weight'])]
            idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
            idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)
            idx_rank = get_rank(idx_profit_by_weight)

        elif o.name == 'max_max_value_by_weight':
            max_profit = np.max(data['value'], 0)
            profit_by_weight = [v / w for v, w in zip(max_profit, data['weight'])]
            idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
            idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)
            idx_rank = get_rank(idx_profit_by_weight)

        # Append to normalized rank features
        idx_rank_array = (1 / n_items) * np.asarray([idx_rank[i] for i in range(n_items)])
        item_features = np.vstack([item_features,
                                   idx_rank_array])

    return item_features


def get_features(opts, data, norm_const=1000):
    # Normalize data
    norm_value = (1 / norm_const) * np.asarray(data['value'])
    norm_weight = (1 / norm_const) * np.asarray(data['weight'])

    # Calculate instance features
    inst_feat = get_instance_features(opts, norm_weight, norm_value)
    inst_feat = inst_feat.reshape((1, -1))
    inst_feat = np.repeat(inst_feat, opts.n, axis=0)
    # print(inst_feat.shape)

    # Calculate item features
    item_feat = get_item_features(data, norm_weight, norm_value)
    item_feat = item_feat.T
    # print(item_feat.shape)

    # Join features and prepare final x
    feat = np.hstack((inst_feat, item_feat))
    # print(feat.shape)

    return feat
