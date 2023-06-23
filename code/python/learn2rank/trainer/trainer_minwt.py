from pathlib import Path

import numpy as np

from learn2rank.utils.data import read_data_from_file
from learn2rank.utils.metrics import eval_learning_metrics
from learn2rank.utils.metrics import eval_order_metrics
from learn2rank.utils.metrics import eval_rank_metrics
from learn2rank.utils.order import get_variable_rank_from_weights
from learn2rank.utils.order import pred_score2order
from .trainer import Trainer


class MinWeightTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None, ps=None, rs=None):
        super(MinWeightTrainer, self).__init__(data, model, cfg, ps, rs)

        self.res_path = Path(self.cfg.res_path[self.cfg.machine])
        self.inst_root_path = self.res_path / 'instances' / cfg.problem.name

        self.min_weight_incb = {'avg_value': 0.0,
                                'avg_value_by_weight': 0.0,
                                'max_value': 0.0,
                                'max_value_by_weight': 0.0,
                                'min_value': 0.0,
                                'min_value_by_weight': 0.0,
                                'weight': -1.0}

        if self.rs is None:
            self.rs = self._get_results_store()
            self.rs['task'] = self.cfg.task
            self.rs['model_name'] = self.cfg.model.name

        if self.ps is None:
            self.ps = self._get_preds_store()

            self.y_tr, names_tr, n_items_tr, self.wt_tr = self._get_split_data(split='train')
            self.ps['train']['names'] = names_tr
            self.ps['train']['n_items'] = n_items_tr

            self.y_val, names_val, n_items_val, self.wt_val = self._get_split_data(split='val')
            self.ps['val']['names'] = names_val
            self.ps['val']['n_items'] = n_items_val

            self.y_test, names_test, n_items_test, self.wt_test = self._get_split_data(split='test')
            self.ps['test']['names'] = names_test
            self.ps['test']['n_items'] = n_items_test

    def run(self):
        self.ps['train']['rank'] = self._get_split_ranks(split='train')
        self.ps['val']['rank'] = self._get_split_ranks(split='val')

        self.rs['train']['learning'] = eval_learning_metrics(self.y_tr, self.ps['train']['rank'], self.wt_tr)
        self.rs['val']['learning'] = eval_learning_metrics(self.y_val, self.ps['val']['rank'], self.wt_val)

        y_tr_order = pred_score2order(self.y_tr)
        self.ps['train']['order'] = pred_score2order(self.ps['train']['rank'])
        y_val_order = pred_score2order(self.y_val)
        self.ps['val']['order'] = pred_score2order(self.ps['val']['rank'])

        # Ranking metrics
        self.rs['train']['ranking'].extend(eval_order_metrics(y_tr_order,
                                                              self.ps['train']['order'],
                                                              self.ps['train']['n_items']))
        self.rs['train']['ranking'].extend(eval_rank_metrics(self.y_tr,
                                                             self.ps['train']['rank'],
                                                             self.ps['train']['n_items']))

        self.rs['val']['ranking'].extend(eval_order_metrics(y_val_order,
                                                            self.ps['val']['order'],
                                                            self.ps['val']['n_items']))
        self.rs['val']['ranking'].extend(eval_rank_metrics(self.y_val,
                                                           self.ps['val']['rank'],
                                                           self.ps['val']['n_items']))
        self.val_tau = -1

        self._save_predictions()
        self._save_results()

    def predict(self, split='test'):
        self.ps[split]['rank'] = self._get_split_ranks(split=split)
        self.ps[split]['order'] = pred_score2order(self.ps[split]['rank'])

        self._save_predictions()
        self._save_results()

    def _get_split_ranks(self, split):
        y_pred_lst = []

        for name in self.ps[split]['names']:
            acronym, _, a, b, pid = name.split("_")
            if acronym == 'kp':
                size = f'{a}_{b}'

                inst = self.inst_root_path / size / split / f'{name}.dat'
                data = read_data_from_file(acronym, inst)
                y_pred = get_variable_rank_from_weights(data, self.min_weight_incb,
                                                        normalized=False)
                y_pred_lst.append(y_pred)

        return y_pred_lst

    def _get_split_data(self, split='train'):
        x, y, wt, names, n_items = [], [], [], [], []

        size = self.cfg.problem.size
        # for size in self.cfg.dataset.size:
        for v in self.data[size][split]:
            _x, _y = v['x'], v['y']
            n_items.append(self.cfg.problem.n_vars)
            names.append(v['name'])
            y.append(_y)

        sample_weights = [1] * len(y)

        return np.asarray(y), names, n_items, sample_weights
