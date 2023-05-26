from pathlib import Path

import numpy as np

from learn2rank.utils.data import read_data_from_file
from learn2rank.utils.metrics import eval_learning_metrics
from learn2rank.utils.metrics import eval_order_metrics
from learn2rank.utils.metrics import eval_rank_metrics
from learn2rank.utils.order import get_variable_rank_from_weights
from learn2rank.utils.order import pred_score2order
from .trainer import Trainer
import pandas as pd
import ast


class SmacAllTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None):
        super(SmacAllTrainer, self).__init__(data, model, cfg)

        self.res_path = Path(self.cfg.res_path[self.cfg.machine])
        self.inst_root_path = self.res_path / 'instances' / cfg.problem.name
        self.traj_path = self.res_path / 'smac_all_output' / cfg.problem.name / cfg.problem.size
        self.traj_path = self.traj_path / cfg.smac_all_path
        self.traj_path = self.traj_path / f'{cfg.problem.acronym}_7_{cfg.problem.size}_0' / 'run_777' / 'traj.json'
        self.incb = ast.literal_eval(self.traj_path.read_text().strip().split('\n')[-1])["incumbent"]


        self.rs = self._get_results_store()
        self.ps = self._get_preds_store()
        self.rs['task'] = self.cfg.task
        self.rs['model_name'] = self.cfg.model.name

    def run(self):
        names_tr, n_items_tr, wt_tr = self._get_split_data(split='train')
        self.ps['tr']['names'] = names_tr
        self.ps['tr']['n_items'] = n_items_tr

        names_val, n_items_val, wt_val = self._get_split_data(split='val')
        self.ps['val']['names'] = names_val
        self.ps['val']['n_items'] = n_items_val

        self.ps['tr']['rank'] = self.predict(names_tr, 'train')
        self.ps['val']['rank'] = self.predict(names_val, 'val')

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
                y_pred = get_variable_rank_from_weights(data, self.incb,
                                                        normalized=False)
                y_pred_lst.append(y_pred)

        return y_pred_lst

    def _get_split_data(self, split='train'):
        x, y, wt, names, n_items = [], [], [], [], []

        size = self.cfg.problem.size
        # for size in self.cfg.dataset.size:
        for v in self.data[size][split]:
            _y = v['y']
            n_items.append(len(_y))
            names.append(v['name'])
            y.append(_y)

        sample_weights = [1] * len(y)

        return names, n_items, sample_weights

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
