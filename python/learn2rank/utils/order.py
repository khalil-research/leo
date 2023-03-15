import json
import random
from operator import itemgetter

import numpy as np

from .const import KnapsackPropertyWeights


def get_order(data):
    order = {
        'max_weight': None,
        'min_weight': None,
        'max_avg_value': None,
        'min_avg_value': None,
        'max_max_value': None,
        'min_max_value': None,
        'max_min_value': None,
        'min_min_value': None,
        'max_avg_value_by_weight': None,
        'max_max_value_by_weight': None,
    }

    for o in order.keys():
        if o == 'max_weight':
            idx_weight = [(i, w) for i, w in enumerate(data['weight'])]
            # print(o, idx_weight)
            idx_weight.sort(key=itemgetter(1), reverse=True)
            order[o] = [i[0] for i in idx_weight]

        elif o == 'min_weight':
            idx_weight = [(i, w) for i, w in enumerate(data['weight'])]
            idx_weight.sort(key=itemgetter(1))
            order[o] = [i[0] for i in idx_weight]

        elif o == 'max_avg_value':
            mean_profit = np.mean(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(mean_profit)]
            idx_profit.sort(key=itemgetter(1), reverse=True)
            order[o] = [i[0] for i in idx_profit]

        elif o == 'min_avg_value':
            mean_profit = np.mean(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(mean_profit)]
            idx_profit.sort(key=itemgetter(1))
            order[o] = [i[0] for i in idx_profit]

        elif o == 'max_max_value':
            max_profit = np.max(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(max_profit)]
            idx_profit.sort(key=itemgetter(1), reverse=True)
            order[o] = [i[0] for i in idx_profit]

        elif o == 'min_max_value':
            max_profit = np.max(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(max_profit)]
            idx_profit.sort(key=itemgetter(1))
            order[o] = [i[0] for i in idx_profit]

        elif o == 'max_min_value':
            min_profit = np.min(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(min_profit)]
            idx_profit.sort(key=itemgetter(1), reverse=True)
            order[o] = [i[0] for i in idx_profit]

        elif o == 'min_min_value':
            min_profit = np.min(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(min_profit)]
            idx_profit.sort(key=itemgetter(1))
            order[o] = [i[0] for i in idx_profit]

        elif o == 'max_avg_value_by_weight':
            mean_profit = np.mean(data['value'], 0)
            profit_by_weight = [v / w for v, w in zip(mean_profit, data['weight'])]
            idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
            idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)
            order[o] = [i[0] for i in idx_profit_by_weight]

        elif o == 'max_max_value_by_weight':
            max_profit = np.max(data['value'], 0)
            profit_by_weight = [v / w for v, w in zip(max_profit, data['weight'])]
            idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
            idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)
            order[o] = [i[0] for i in idx_profit_by_weight]

    return order


def get_weighted_order(opts, data, weighted_ordering_dict):
    orders = get_order(data)
    num_items = opts.n

    # for o in ordering_lst:
    #     print(orders[o])
    #     print()

    scores = [0] * num_items
    for o in orders.keys():
        for rank, item_id in enumerate(orders[o]):
            item_score = num_items - rank
            scores[item_id] += weighted_ordering_dict[o] * item_score

    # for i, s in enumerate(scores):
    #     print(i, np.round(s, 4))

    new_order = []
    for item_id, score in enumerate(scores):
        new_order.append((item_id, score))
    new_order.sort(key=itemgetter(1), reverse=True)
    new_order = [k for k, v in new_order]

    return new_order


def get_variable_score_from_weights(data, property_weights):
    """Given variables score based on property weights"""
    weight, value = np.asarray(data['weight']), np.asarray(data['value'])
    n_items = weight.shape[0]
    value_mean = np.mean(value, axis=0)
    value_max = np.max(value, axis=0)
    value_min = np.min(value, axis=0)

    # Normalized variable score computation
    scores = np.zeros(n_items)
    for fk, fv in property_weights.items():
        _norm_scores = np.zeros(n_items)
        if fk == 'weight':
            _norm_scores = weight

        elif fk == 'avg_value':
            _norm_scores = value_mean

        elif fk == 'max_value':
            _norm_scores = value_max

        elif fk == 'min_value':
            _norm_scores = value_min

        elif fk == 'avg_value_by_weight':
            _norm_scores = (value_mean / weight)

        elif fk == 'max_value_by_weight':
            _norm_scores = (value_max / weight)

        elif fk == 'min_value_by_weight':
            _norm_scores = (value_min / weight)

        elif fk == 'label':
            _norm_scores = np.arange(1, n_items + 1)[::-1]

        _norm_scores = _norm_scores / np.sum(_norm_scores)
        assert _norm_scores.shape[0] == n_items
        assert np.round(_norm_scores.sum()) == 1
        scores += fv * _norm_scores

    return scores


def get_variable_order_from_weights(data, property_weights):
    """Returns array of variable order
    For example: [2, 1, 0]
    Here 2 is the index of item which should be used first to create the BDD
    """
    n_items = len(data['weight'])
    scores = get_variable_score_from_weights(data, property_weights)

    idx_score = [(i, v) for i, v in zip(np.arange(n_items), scores)]
    idx_score.sort(key=itemgetter(1), reverse=True)

    order = [i for i, v in idx_score]

    return order, idx_score


def get_variable_rank_from_weights(data, property_weights, normalized=True):
    """Returns array of variable ranks
    For example: [2, 1, 0]
    Item 0 must be used third to construct the BDD
    """
    n_items = len(data['weight'])

    _, idx_score_desc = get_variable_order_from_weights(data, property_weights)

    variable_rank = np.zeros(n_items)
    for rank, (i, _) in enumerate(idx_score_desc):
        variable_rank[i] = rank
    if normalized:
        variable_rank /= n_items

    return variable_rank


def property_weight_dict2array(pw_dict, cast_to_numpy=False):
    lst = []
    for i in range(len(KnapsackPropertyWeights)):
        lst.append(pw_dict[KnapsackPropertyWeights(i).name])

    if cast_to_numpy:
        lst = np.asarray(lst)

    return lst


def score2order(scores):
    if type(scores) == list:
        scores = np.asarray(scores)

    assert len(scores.shape) <= 2
    if len(scores.shape) == 1:
        scores = scores.reshape(1, -1)

    orders = []
    for _scores in scores:
        idx_rank = []
        for item, score in enumerate(_scores):
            idx_rank.append((item, score))

        idx_rank.sort(key=itemgetter(1))
        order = [int(i[0]) for i in idx_rank]
        orders.append(order)

    return np.asarray(orders)


def get_static_orders(data, order_type=None):
    if order_type == None:
        order = {
            'max_weight': None,
            'min_weight': None,
            'max_avg_profit': None,
            'min_avg_profit': None,
            'max_max_profit': None,
            'min_max_profit': None,
            'max_min_profit': None,
            'min_min_profit': None,
            'max_avg_profit_by_weight': None,
            'max_max_profit_by_weight': None,
        }
    else:
        if type(order_type) == str:
            order = {order_type: None}
        elif type(order_type) == list:
            order = {ot: None for ot in order_type}

    n_items = len(data['weight'])
    for o in order.keys():
        if o == 'max_weight':
            idx_weight = [(i, w) for i, w in enumerate(data['weight'])]
            # print(o, idx_weight)
            idx_weight.sort(key=itemgetter(1), reverse=True)
            order[o] = [i[0] for i in idx_weight]

        elif o == 'min_weight':
            idx_weight = [(i, w) for i, w in enumerate(data['weight'])]
            idx_weight.sort(key=itemgetter(1))
            order[o] = [i[0] for i in idx_weight]

        elif o == 'max_avg_profit':
            mean_profit = np.mean(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(mean_profit)]
            idx_profit.sort(key=itemgetter(1), reverse=True)
            order[o] = [i[0] for i in idx_profit]

        elif o == 'min_avg_profit':
            mean_profit = np.mean(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(mean_profit)]
            idx_profit.sort(key=itemgetter(1))
            order[o] = [i[0] for i in idx_profit]

        elif o == 'max_max_profit':
            max_profit = np.max(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(max_profit)]
            idx_profit.sort(key=itemgetter(1), reverse=True)
            order[o] = [i[0] for i in idx_profit]

        elif o == 'min_max_profit':
            max_profit = np.max(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(max_profit)]
            idx_profit.sort(key=itemgetter(1))
            order[o] = [i[0] for i in idx_profit]

        elif o == 'max_min_profit':
            min_profit = np.min(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(min_profit)]
            idx_profit.sort(key=itemgetter(1), reverse=True)
            order[o] = [i[0] for i in idx_profit]

        elif o == 'min_min_profit':
            min_profit = np.min(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(min_profit)]
            idx_profit.sort(key=itemgetter(1))
            order[o] = [i[0] for i in idx_profit]

        elif o == 'max_avg_profit_by_weight':
            mean_profit = np.mean(data['value'], 0)
            profit_by_weight = [v / w for v, w in zip(mean_profit, data['weight'])]
            idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
            idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)
            # print(idx_profit_by_weight)
            order[o] = [i[0] for i in idx_profit_by_weight]

        elif o == 'max_max_profit_by_weight':
            max_profit = np.max(data['value'], 0)
            profit_by_weight = [v / w for v, w in zip(max_profit, data['weight'])]
            idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
            idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)
            # print(idx_profit_by_weight)
            order[o] = [i[0] for i in idx_profit_by_weight]

    return order


def get_smac_path(cfg, resource_path, pid):
    path = resource_path / 'smac_output' / cfg.problem.name
    if cfg.smac_dir is None:
        path = path / cfg.problem.size / cfg.split
    else:
        path = path / cfg.smac_dir / cfg.problem.size / cfg.split

    if cfg.problem.name == 'knapsack':
        path = path / f'kp_{cfg.seed.opt}_{cfg.problem.size}_{pid}' / f'run_{cfg.seed.smac}'
    elif cfg.problem.name == 'setcovering' or cfg.problem.name == 'setpacking':
        path = path / f'bp_{cfg.seed.opt}_{cfg.problem.size}_{pid}' / f'run_{cfg.seed.smac}'
    else:
        raise ValueError('Invalid problem type!')

    return path


def get_smac_all_path(cfg, resource_path):
    path = resource_path / 'smac_all_output' / cfg.problem.name

    assert cfg.smac_all_dir is not None
    path = path / cfg.smac_all_dir / cfg.problem.size

    if cfg.problem.name == 'knapsack':
        path = path / f'kp_{cfg.seed.opt}_{cfg.problem.size}_0' / f'run_{cfg.seed.smac}'
    elif cfg.problem.name == 'setcovering' or cfg.problem.name == 'setpacking':
        path = path / f'bp_{cfg.seed.opt}_{cfg.problem.size}_0' / f'run_{cfg.seed.smac}'
    else:
        raise ValueError('Invalid problem type!')

    return path


def get_baseline_order(data, cfg, resource_path, pid):
    orders = []
    if cfg.order_type == 'min_weight':
        order = get_static_orders(data, order_type='min_weight')
        orders.append(order['min_weight'])

    if cfg.order_type == 'max_weight':
        order = get_static_orders(data, order_type='max_weight')
        orders.append(order['max_weight'])

    elif cfg.order_type == 'canonical':
        orders.append(list(range(len(data['weight']))))

    elif cfg.order_type == 'rand':
        seeds = [13, 444, 1212, 1003, 7517]
        for s in seeds:
            random_order = list(range(len(data['weight'])))
            random.seed(s)
            random.shuffle(random_order)
            orders.append(random_order)

    elif cfg.order_type == 'smac':
        run_path = get_smac_path(cfg, resource_path, pid)
        traj_path = run_path / 'traj.json'
        if traj_path.exists():
            # Get property weight
            traj = traj_path.open('r')
            lines = traj.readlines()
            property_weight = json.loads(lines[-1])
            order, _ = get_variable_order_from_weights(data, property_weight['incumbent'])
            orders.append(order)

    elif cfg.order_type == 'smac_all':
        run_path = get_smac_all_path(cfg, resource_path)
        traj_path = run_path / 'traj.json'
        if traj_path.exists():
            # Get property weight
            traj = traj_path.open('r')
            lines = traj.readlines()
            property_weight = json.loads(lines[-1])
            order, _ = get_variable_order_from_weights(data, property_weight['incumbent'])
            orders.append(order)

    return orders


def make_result_column(problem, size, split, pid, order_type, result, run_id=0):
    col = [problem, size, split, pid, order_type, run_id]
    col.extend(result)

    return col
