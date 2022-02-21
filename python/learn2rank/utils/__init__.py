import pickle as pkl
import resource
from itertools import product
from operator import itemgetter
from subprocess import Popen, PIPE, TimeoutExpired

import numpy as np
import scipy as sp
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.data import Dataset, DataLoader

from .const import numpy_dataset_paths
from .featurizer import get_features
from .featurizer import get_instance_features
from .featurizer import get_item_features
from .order import get_incumbent_lst
from .order import get_variable_rank_from_weights

# Maximal virtual memory for subprocesses (in bytes).
MAX_VIRTUAL_MEMORY = 1 * 1024 * 1024 * 1024  # 1 GB


class PointwiseVariableRankRegressionDataset(Dataset):
    def __init__(self, dict_dataset_path, device):
        self.dict_dataset_np = pkl.load(open(dict_dataset_path, 'rb'))

        self.num_instances = len((self.dict_dataset.keys()))
        # Var feat shape: Num features x Num items
        self.num_items = self.dict_dataset[0]['var_feat'].shape[1]
        self.idx2item = {idx: (inst, item)
                         for idx, inst, item in enumerate(product(range(self.num_instances),
                                                                  range(self.num_items)))}

    def __len__(self):
        return self.num_instances * self.num_items

    def __getitem__(self, idx):
        inst, item = self.idx2item[idx]

        return 1


class WeightRegressionDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, item):
        return 1


class VariableFeaturesDataset(Dataset):
    def __init__(self, dict_dataset_path, device, usage=''):
        dict_dataset = pkl.load(open(dict_dataset_path, 'rb'))
        self.num_instances = len((dict_dataset.keys()))
        # Var feat shape: Num features x Num items
        self.num_items = dict_dataset[0]['var_feat'].shape[1]
        self.inst_item_map = list(product(range(self.num_instances), range(self.num_items)))
        self.usage

    def __len__(self):
        return self.num_instances * self.num_items

    def __getitem__(self, idx):
        inst, item = self.inst_item_map[idx]

        return 1


def get_rank(sorted_data):
    idx_rank = {}
    for rank, item in enumerate(sorted_data):
        idx_rank[item[0]] = rank

    return idx_rank


def get_order_from_rank(ranks):
    idx_rank = []
    for item, rank in enumerate(ranks):
        idx_rank.append((item, rank))

    idx_rank.sort(key=itemgetter(1))
    order = [int(i[0]) for i in idx_rank]

    return order


def read_from_file(p, filepath):
    data = {'value': [], 'weight': [], 'capacity': 0}

    with open(filepath, 'r') as fp:
        fp.readline()
        fp.readline()

        for _ in range(p):
            data['value'].append(list(map(int, fp.readline().split())))

        data['weight'].extend(list(map(int, fp.readline().split())))

        data['capacity'] = int(fp.readline().split()[0])

    return data


def get_static_orders(data):
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


def limit_virtual_memory():
    # Maximal virtual memory for subprocesses (in bytes).
    # MAX_VIRTUAL_MEMORY = 1 * 1024 * 1024 * 1024  # 1 GB
    global MAX_VIRTUAL_MEMORY

    # The tuple below is of the form (soft limit, hard limit). Limit only
    # the soft part so that the limit can be increased later (setting also
    # the hard limit would prevent that).
    # When the limit cannot be changed, setrlimit() raises ValueError.
    resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY, MAX_VIRTUAL_MEMORY))


def run_bdd_builder(instance, order, time_limit=60, mem_limit=2):
    # Prepare the call string to binary
    order_string = " ".join(map(str, order))
    cmd = f"./multiobj {instance} {len(order)} {order_string}"
    # Maximal virtual memory for subprocesses (in bytes).
    global MAX_VIRTUAL_MEMORY
    MAX_VIRTUAL_MEMORY = mem_limit * (1024 ** 3)

    status = "SUCCESS"
    runtime = 0
    try:
        io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE, preexec_fn=limit_virtual_memory)
        # Call target algorithm with cutoff time
        (stdout_, stderr_) = io.communicate(timeout=time_limit)

        # Decode and parse output
        stdout, stderr = stdout_.decode('utf-8'), stderr_.decode('utf-8')
        if len(stdout) and "Solved" in stdout:
            # Sum the last three floating points to calculate the total time
            # This is binary dependent and can change
            runtime = np.sum(list(map(float, stdout.strip().split(',')[-3:])))
        else:
            # If the instance is not solved successfully on the cluster, we either hit the
            # runtime limit or memory limit. In either of the two cases, we will not be
            # allowed to run more instances. Hence, we stop the parameter optimization
            # process using the ABORT signal
            status = "ABORT"
            runtime = time_limit
    except TimeoutExpired:
        status = "TIMEOUT"
        runtime = time_limit

    return status, runtime


def load_split(dataset, split='train'):
    x = [np.load(_dataset['X']) for _dataset in dataset[split]]
    y = [np.load(_dataset['Y']) for _dataset in dataset[split]]
    y_shape = [_y.shape for _y in y]

    return x, y, y_shape


def flatten_data(X, Y, loss_weights_type=0):
    x_flat, y_flat, weights_flat = [], [], []
    for _X, _Y in zip(X, Y):
        for _sampleX, _sampleY in zip(_X, _Y):
            scale = np.max(_sampleY) - np.min(_sampleY)
            for _itemX, _itemY in zip(_sampleX, _sampleY):
                _weight = 1
                if loss_weights_type == 1:
                    """Linearly decreasing"""
                    _weight = 1 - (_itemY / scale)
                elif loss_weights_type == 2:
                    """Exponentially decreasing"""
                    _weight = np.exp(-((_itemY / scale) * 5))
                weights_flat.append(_weight)

                x_flat.append(_itemX)
                y_flat.append(_itemY)

    x_flat, y_flat = np.asarray(x_flat), np.asarray(y_flat)
    weights_flat = np.asarray(weights_flat)
    print(x_flat.shape, y_flat.shape)

    return x_flat, y_flat, weights_flat


def unflatten_data(y, y_shape):
    Y_out = []

    if y is not None:
        i = 0
        for (num_samples, num_items) in y_shape:
            y_out = []
            for j in range(num_samples):
                y_out.append(y[i: i + num_items])
                i = i + num_items

            Y_out.append(np.asarray(y_out))

    return Y_out


def get_unnormalized_variable_rank(y_norm):
    unnorm_ranks = {}
    for split in y_norm.keys():
        if y_norm[split] is None:
            continue

        unnorm_ranks[split] = []
        for _dataset in y_norm[split]:
            unnorm_ranks_dataset = []
            for _norm_pred in _dataset:
                item_norm_rank = [(idx, pred) for idx, pred in enumerate(_norm_pred)]
                # Sort ascending. Smaller the rank higher the precedence
                item_norm_rank.sort(key=itemgetter(1))
                variable_ranks = np.zeros_like(_norm_pred)
                for rank, (item, _) in zip(range(_norm_pred.shape[0]), item_norm_rank):
                    variable_ranks[item] = rank

                unnorm_ranks_dataset.append(variable_ranks)

            unnorm_ranks[split].append(np.asarray(unnorm_ranks_dataset))

    return unnorm_ranks


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
