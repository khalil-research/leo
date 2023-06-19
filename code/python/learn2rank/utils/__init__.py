import hashlib
import random
from operator import itemgetter

import numpy as np
import torch

from .const import numpy_dataset_paths
from .data import read_data_from_file
from .metrics import eval_learning_metrics, eval_order_metrics
from .order import get_variable_rank_from_weights
from .order import property_weight_dict2array


class Factory:
    def __init__(self):
        self._members = {}

    def register_member(self, key, member_cls):
        self._members[key] = member_cls

    def create(self, key, **kwargs):
        member_cls = self._members.get(key)
        if member_cls is None:
            raise ValueError(key)

        return member_cls(**kwargs)


def hashit(x):
    return hashlib.blake2s(x.encode('utf-8'), digest_size=32).hexdigest()


def set_machine(cfg):
    if 'machine' in cfg:
        import os
        cfg.machine = os.environ.get('machine')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def get_unnormalized_variable_rank(y_norm):
    unnorm_ranks = []

    for norm_pred in y_norm:
        item_norm_rank = [(idx, pred) for idx, pred in enumerate(norm_pred)]
        # Sort ascending. Smaller the rank higher the precedence
        item_norm_rank.sort(key=itemgetter(1))
        variable_ranks = np.zeros_like(norm_pred)
        for rank, (item, _) in zip(range(norm_pred.shape[0]), item_norm_rank):
            variable_ranks[item] = rank
        unnorm_ranks.append(variable_ranks)

    return unnorm_ranks
