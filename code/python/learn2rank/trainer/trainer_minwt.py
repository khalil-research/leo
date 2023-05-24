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
    def __init__(self, data=None, model=None, cfg=None):
        super(MinWeightTrainer, self).__init__(data, model, cfg)

        self.res_path = Path(self.cfg.res_path[self.cfg.machine])
        self.inst_root_path = self.res_path / 'instances' / cfg.problem.name

        self.min_weight_incb = {'avg_value': 0.0,
                                'avg_value_by_weight': 0.0,
                                'max_value': 0.0,
                                'max_value_by_weight': 0.0,
                                'min_value': 0.0,
                                'min_value_by_weight': 0.0,
                                'weight': -1.0}

        self.rs = self._get_results_store()
        self.ps = self._get_preds_store()
        self.rs['task'] = self.cfg.task
        self.rs['model_name'] = self.cfg.model.name

    def run(self):
        y_tr, names_tr, n_items_tr, wt_tr = self._get_split_data(split='train')
        self.ps['tr']['names'] = names_tr
        self.ps['tr']['n_items'] = n_items_tr

        y_val, names_val, n_items_val, wt_val = self._get_split_data(split='val')
        self.ps['val']['names'] = names_val
        self.ps['val']['n_items'] = n_items_val

        self.ps['tr']['rank'] = self.predict(names_tr, 'train')
        self.ps['val']['rank'] = self.predict(names_val, 'val')

        self.rs['tr']['learning'] = eval_learning_metrics(y_tr, self.ps['tr']['rank'], wt_tr)
        self.rs['val']['learning'] = eval_learning_metrics(y_val, self.ps['val']['rank'], wt_val)

        y_tr_order = pred_score2order(y_tr)
        self.ps['tr']['order'] = pred_score2order(self.ps['tr']['rank'])
        y_val_order = pred_score2order(y_val)
        self.ps['val']['order'] = pred_score2order(self.ps['val']['rank'])

        # Ranking metrics
        self.rs['tr']['ranking'].extend(eval_order_metrics(y_tr_order,
                                                           self.ps['tr']['order'],
                                                           n_items_tr))
        self.rs['tr']['ranking'].extend(eval_rank_metrics(y_tr, self.ps['tr']['rank'], n_items_tr))

        self.rs['val']['ranking'].extend(eval_order_metrics(y_val_order,
                                                            self.ps['val']['order'],
                                                            n_items_val))
        self.rs['val']['ranking'].extend(eval_rank_metrics(y_val, self.ps['val']['rank'], n_items_val))

        self._save_predictions()
        self._save_results()

    def predict(self, names, split):
        y_pred_lst = []
        for name in names:
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
            n_items.append(len(_y))
            names.append(v['name'])
            y.append(_y)

        sample_weights = [1] * len(y)

        return np.asarray(y), names, n_items, sample_weights

    @staticmethod
    def _get_results_store():
        return {
            'task': None,
            'model_name': None,
            'tr': {
                'learning': {},
                'ranking': [],
            },
            'val': {
                'learning': {},
                'ranking': []
            },
            'test': {
                'learning': {},
                'ranking': []
            },
            'time': {
                'train': 0.0,
                'test': 0.0,
                'eval': 0.0
            }
        }

    @staticmethod
    def _get_preds_store():
        return {
            'tr': {
                'names': [],
                'n_items': [],
                'score': [],
                'rank': [],
                'order': []
            },
            'val': {
                'names': [],
                'n_items': [],
                'score': [],
                'rank': [],
                'order': []
            }
        }
