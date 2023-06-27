import ast
import json
import random
from operator import itemgetter

import numpy as np

from .const import KnapsackPropertyWeights
import pandas as pd


def property_weight_dict2array(pw_dict, cast_to_numpy=False):
    lst = []
    for i in range(len(KnapsackPropertyWeights)):
        lst.append(pw_dict[KnapsackPropertyWeights(i).name])

    if cast_to_numpy:
        lst = np.asarray(lst)

    return lst


def _score2order(scores, reverse=False):
    """
    Order variables based on its score
    If reverse is False (default), then lower score will lead to a
    higher place high in the order
    If reverse is True, then higher score will lead to a higher place
    in the order. Used with SVMRank.

    [80, -10, 50, 100], reverse=False ==> [1, 2, 0, 3]
    [80, -10, 50, 100], reverse=True ==> [3, 0, 2, 1]
    """
    orders = []
    for _scores in scores:
        var_rank_score = []
        for var, score in enumerate(_scores):
            var_rank_score.append((var, score))

        var_rank_score.sort(key=itemgetter(1), reverse=reverse)
        order = [int(var) for (var, _) in var_rank_score]
        orders.append(order)

    return orders


def _score2rank(scores, reverse=False, high_to_low=False, normalized=False):
    """Rank variables based on its score.
    If reverse is False (default), then lower score will lead to a
    higher rank
    If reverse is True, then lower score will lead to a lower rank

    [80, -10, 50, 100], reverse=False ==> [2, 0, 1, 3]
    [80, -10, 50, 100], reverse=True ==> [1, 3, 2, 0]
    """
    ranks = []
    for _scores in scores:
        var_rank_score = []
        for var, score in enumerate(_scores):
            var_rank_score.append((var, score))

        var_rank_score.sort(key=itemgetter(1), reverse=reverse)

        n_items = len(_scores)
        _ranks = [0] * n_items
        for rank, (var, _) in enumerate(var_rank_score):
            _ranks[int(var)] = rank if high_to_low is False else n_items - rank

        if normalized:
            _ranks = np.array(_ranks) / n_items

        ranks.append(_ranks)

    return ranks


def get_static_orders(data, order_type=None, property_weights=None):
    if order_type is None:
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
    elif type(order_type) == str:
        order = {order_type: None}
    elif type(order_type) == list:
        order = {ot: None for ot in order_type}
    else:
        raise ValueError('Invalid type(order_type)...')

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

        elif o == 'max_min_profit_by_weight':
            min_profit = np.min(data['value'], 0)
            profit_by_weight = [v / w for v, w in zip(min_profit, data['weight'])]
            idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
            idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)
            # print(idx_profit_by_weight)
            order[o] = [i[0] for i in idx_profit_by_weight]

        elif o == 'smac_instance' or o == 'smac_dataset':
            assert property_weights is not None
            order[o] = get_variable_order(data=data, property_weights=property_weights, reverse=True)[0]

    return order


def get_weighted_order(opts, data, weighted_ordering_dict):
    orders = get_static_orders(data)
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


def get_variable_order(data=None, property_weights=None, scores=None, reverse=False):
    """
    Get the variable order by providing either `data` and `property_weights` or `scores`.
    If `data` and `property_weights` are provided, we first calculate the scores of that instance and put it in a list.
    Once we have `scores`, obtained either as input or calculated from `data` and `property_weights`, we calculate the
    order based on `reverse`.

    If `reverse` is False/True then the items are sorted in ascending/descending order.

    Parameters
    ----------
    data : dict or None
        A dictionary containing the problem instance
    property_weights : dict or None
        A dictionary containing the weights of different properties
    scores : list[list] or None
        A list of list containing scores of items of different instances
    reverse : bool, default is False
        Flag to select sorting mechanism; False -> Ascending; True -> Descending

    Returns
    -------
    A list of list containing the order of items of different instances.
    If the function was invoked using `data` and `property_weights`, the parent list will contain only one child list.
    """
    scores = [get_variable_score_from_weights(data, property_weights)] if scores is None else scores
    orders = _score2order(scores, reverse=reverse)

    return orders


def get_variable_rank(data=None, property_weights=None, scores=None, reverse=False, high_to_low=False,
                      normalized=False):
    """
    Get the variable rank by providing either `data` and `property_weights` or `scores`.
    If `data` and `property_weights` are provided, we first calculate the scores of that instance and put it in a list.
    Once we have `scores`, obtained either as input or calculated from `data` and `property_weights`, we calculate ts
    rank based on `reverse`, `high_to_low` and `normalized` parameters.

    If `reverse` is False/True then the items are sorted in ascending/descending order.
    If `high_to_low` is False/True then the first element sorted based on score get a rank 0/N.
    If `normalizes` is False/True then the rank is not divided/get divided by the number of items.

    Parameters
    ----------
    data : dict or None
        A dictionary containing the problem instance
    property_weights : dict or None
        A dictionary containing the weights of different properties
    scores : list[list] or None
        A list of list containing scores of items of different instances
    reverse : bool, default is False
        Flag to select sorting mechanism; False -> Ascending; True -> Descending
    high_to_low : bool, default is False
        Flag to select ranking mechanism; False -> First element gets rank 0; True -> First element get rank N
    normalized : bool, default is False
        Flag to select normalization; False -> Do not normalize; True -> Normalize

    Returns
    -------
    A list of list containing the rank of items of different instances.
    If the function was invoked using `data` and `property_weights`, the parent list will contain only one child list.
    """
    scores = [get_variable_score_from_weights(data, property_weights)] if scores is None else scores
    ranks = _score2rank(scores, reverse=reverse, high_to_low=high_to_low, normalized=normalized)

    return ranks


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

    elif cfg.order_type == 'lex':
        orders.append(list(range(len(data['weight']))))

    elif cfg.order_type == 'rand':
        seeds = [13, 444, 1212, 1003, 7517]
        for s in seeds:
            random_order = list(range(len(data['weight'])))
            random.seed(s)
            random.shuffle(random_order)
            orders.append(random_order)

    elif cfg.order_type == 'smac':
        label_path = resource_path / 'labels' / cfg.problem.name / cfg.problem.size / f'label_{cfg.problem.size}.csv'
        df = pd.read_csv(label_path)
        incb = ast.literal_eval(df[df['pid'] == pid].iloc[0, 'incb'])
        order = get_variable_order(data=data, property_weights=incb, reverse=True)[0]
        orders.append(order)

    elif cfg.order_type == 'smac_all':
        run_path = get_smac_all_path(cfg, resource_path)
        traj_path = run_path / 'traj.json'
        if traj_path.exists():
            # Get property weight
            traj = traj_path.open('r')
            lines = traj.readlines()
            property_weight = json.loads(lines[-1])
            order = get_variable_order(data=data, property_weights=property_weight['incumbent'], reverse=True)
            orders.append(order)

    return orders


def make_result_column(problem, size, split, pid, order_type, result, run_id=0):
    col = [problem, size, split, pid, order_type, run_id]
    col.extend(result)

    return col
