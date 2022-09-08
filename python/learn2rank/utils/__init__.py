import random
from operator import itemgetter

import numpy as np
import torch

from .const import numpy_dataset_paths
from .data import read_data_from_file
from .metrics import eval_learning_metrics, eval_rank_metrics
from .order import get_incumbent_lst
from .order import get_incumbent_lst
from .order import get_variable_rank_from_weights


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
<<<<<<< Updated upstream


def eval_learning_metrics(orig, pred, weights):
    mse = mean_squared_error(orig, pred)
    mae = mean_absolute_error(orig, pred)
    r2 = r2_score(orig, pred, sample_weight=weights)

    print('MSE: ', mse)
    print('MAE: ', mae)
    print('R2: ', r2)

    return {'MSE': mse, 'R2': r2, 'MAE': mae}


def eval_rank_metrics(orig, pred):
    for k in orig.keys():
        if orig[k] is None:
            continue

        print(f'{k} rank metrics...')
        corrs = []
        ps = []
        top_10_common, top_5_common = [], []
        top_10_same, top_5_same = [], []
        top_10_penalty, top_5_penalty = [], []
        for odata, pdata in zip(orig[k], pred[k]):
            # Spearman rank correlation
            for oranks, pranks in zip(odata, pdata):
                corr, p = sp.stats.spearmanr(oranks, pranks)
                corrs.append(corr)
                ps.append(p)

                oorder = np.asarray(get_order_from_rank(oranks))
                porder = np.asarray(get_order_from_rank(pranks))

                # Top 10 accuracy
                top_10_common.append(len(set(oorder[:10]).intersection(set(porder[:10]))))
                top_10_same.append(np.sum(oorder[:10] == porder[:10]))

                _penalties = []
                for j in range(10):
                    _penalties.append(np.abs(j - np.where(porder == oorder[j])[0][0]))
                top_10_penalty.append(np.mean(_penalties))

                # Top 5 accuracy
                top_5_common.append(len(set(oorder[:5]).intersection(set(porder[:5]))))
                top_5_same.append(np.sum(oorder[:5] == porder[:5]))

                _penalties = []
                for j in range(5):
                    _penalties.append(np.abs(j - np.where(porder == oorder[j])[0][0]))
                top_5_penalty.append(np.mean(_penalties))

        assert len(top_10_penalty) == len(top_5_penalty)

        print('Correlation    :', np.mean(corrs), np.std(corrs))
        print('p-value        :', np.mean(ps), np.std(ps))
        print('Top 10 Common  :', np.mean(top_10_common))
        print('Top 10 Same    :', np.mean(top_10_same))
        print('Top 10 Penalty :', np.mean(top_10_penalty))
        print('Top 5 Common   :', np.mean(top_5_common))
        print('Top 5 Same     :', np.mean(top_5_same))
        print('Top 5 Penalty  :', np.mean(top_5_penalty))

        print()


def get_tensor_datasets(args, dataset_paths, device):
    tr_dataset = VariableFeaturesDataset(dataset_paths['train'], device)
    val_dataset = VariableFeaturesDataset(dataset_paths['val'], device)
    test_dataset = VariableFeaturesDataset(dataset_paths['test'], device)


def get_dataloaders(args, tr_dataset, val_dataset, test_dataset):
    tr_loader = DataLoader(tr_dataset, shuffle=True, batch_size=args.bs)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=len(val_dataset))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset))

    return tr_loader, val_loader, test_loader
=======
>>>>>>> Stashed changes
