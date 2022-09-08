import numpy as np
import scipy as sp
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from .order import get_order_from_rank


def eval_learning_metrics(orig, pred, weights):
    mse = mean_squared_error(orig, pred)
    mae = mean_absolute_error(orig, pred)
    r2 = r2_score(orig, pred, sample_weight=weights)

    print('MSE: ', mse)
    print('MAE: ', mae)
    print('R2: ', r2)

    return {'MSE': mse, 'R2': r2, 'MAE': mae}


def eval_rank_metrics(orig, pred):
    if orig is None:
        return {}

    corrs = []
    ps = []
    top_10_common, top_5_common = [], []
    top_10_same, top_5_same = [], []
    top_10_penalty, top_5_penalty = [], []
    for oranks, pranks in zip(orig, pred):
        # Spearman rank correlation
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

    results = {
        'correlation': (np.mean(corrs), np.std(corrs)),
        'p_value': (np.mean(ps), np.std(ps)),
        'top_10_common': np.mean(top_10_common),
        'top_10_penalty': np.mean(top_10_penalty),
        'top_10_same': np.mean(top_10_same),
        'top_5_common': np.mean(top_5_common),
        'top_5_penalty': np.mean(top_5_penalty),
        'top_5_same': np.mean(top_5_same)
    }

    print('Correlation    :', results['correlation'])
    print('p-value        :', results['p_value'])
    print('Top 10 Common  :', results['top_10_common'])
    print('Top 10 Penalty :', results['top_10_penalty'])
    print('Top 10 Same    :', results['top_10_same'])
    print('Top 5 Common   :', results['top_5_common'])
    print('Top 5 Penalty  :', results['top_5_penalty'])
    print('Top 5 Same     :', results['top_5_same'])

    return results
