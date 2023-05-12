import logging
import pickle
import time

import os
from pathlib import Path

from learn2rank.utils.metrics import eval_order_metrics
from learn2rank.utils.metrics import eval_rank_metrics
from learn2rank.utils.order import pred_score2order
from learn2rank.utils.order import pred_score2rank
from .trainer import Trainer
from sklearn.datasets import load_svmlight_file
log = logging.getLogger(__name__)


class XGBoostTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None):
        super().__init__(data, model, cfg)
        self.res_path = Path(self.cfg.res_path[self.cfg.machine])

        # Load files
        self.data = Path(data)
        self.train_data_file = self.data / f'{self.cfg.problem.name}_dataset_pair_svmrank_train.dat'
        self.val_data_file = self.data / f'{self.cfg.problem.name}_dataset_pair_svmrank_val.dat'
        self.train_n_items_file = self.data / f'{self.cfg.problem.name}_n_items_pair_svmrank_train.dat'
        self.val_n_items_file = self.data / f'{self.cfg.problem.name}_n_items_pair_svmrank_val.dat'
        # self.train_names_file = self.data / f'{self.cfg.problem.name}_names_pair_svmrank_train.dat'
        # self.val_names_file = self.data / f'{self.cfg.problem.name}_names_pair_svmrank_val.dat'
        # Process files
        self.x_train, self.y_train = load_svmlight_file(self.train_data_file)
        self.x_val, self.y_val = load_svmlight_file(self.val_data_file)
        self.group_train = list(map(int, self.train_n_items_file.read_text().split('\n')))
        self.group_val = list(map(int, self.train_n_items_file.read_text().split('\n')))

        self.x_train_uf, self.y_train_uf = self.unflatten_data(self.x_train, self.y_train, self.group_train)
        self.x_val_uf, self.y_val_uf = self.unflatten_data(self.x_val, self.y_val, self.group_val)

        self.ps = self._get_preds_store()
        self.rs = self._get_results_store()

    def run(self):
        self.model.fit(self.x_train, self.y_train, self.group_train,
                       eval_set=[(self.x_val, self.y_val)],
                       eval_group=self.group_val)

        # Train pred
        for x in self.x_train_uf:
            self.ps['tr']['score'].append(self.model.predict(x))

        # Val pred
        for x in self.x_val_uf:
            self.ps['val']['score'].append(self.model.predict(x))

        # Eval learning metrics
        log.info(f"* {self.cfg.model.name} Results")
        log.info("** Train learning metrics:")
        self.rs['tr']['learning'] = self.eval_learning_metrics(split='train')
        log.info("** Validation learning metrics:")
        self.rs['val']['learning'] = self.eval_learning_metrics(split='val')

        # Transform scores to order
        self.ps['tr']['order'] = pred_score2order(self.ps['tr']['score'], reverse=True)
        self.ps['val']['order'] = pred_score2order(self.ps['val']['score'], reverse=True)

        # Transform scores to ranks
        self.ps['tr']['rank'] = pred_score2rank(self.ps['tr']['score'], reverse=True)
        self.ps['val']['rank'] = pred_score2rank(self.ps['val']['score'], reverse=True)

        # Eval rank predictions
        log.info("** Train order metrics:")
        self.rs['tr']['ranking'].extend(eval_order_metrics(self.y_train_uf,
                                                           self.ps['tr']['order'],
                                                           self.ps['tr']['n_items']))
        self.rs['tr']['ranking'].extend(eval_rank_metrics(self.y_train_uf,
                                                          self.ps['tr']['rank'],
                                                          self.ps['tr']['n_items']))

        log.info("** Val order metrics:")
        self.rs['val']['ranking'].extend(eval_order_metrics(self.y_val_uf,
                                                            self.ps['val']['order'],
                                                            self.ps['val']['n_items']))
        self.rs['val']['ranking'].extend(eval_rank_metrics(self.y_val_uf,
                                                           self.ps['val']['rank'],
                                                           self.ps['val']['n_items']))

        log.info(f"  {self.cfg.model.name} train time: {self.rs['time']['train']} \n")

        self._save_predictions()
        self._save_results()

    @staticmethod
    def unflatten_data(x, y, group):
        x_unflat, y_unflat = [], []

        i = 0
        for g in group:
            x_unflat.append(x[i: i+g])
            y_unflat.append(y[i: i+g])
            i += g

        return x_unflat, y_unflat

    def eval_learning_metrics(self):
        pass

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

    @staticmethod
    def _get_results_store():
        return {
            'tr': {
                'learning': {'mse': None, 'r2': None, 'mae': None, 'mape': None},
                'ranking': [],
            },
            'val': {
                'learning': {'mse': None, 'r2': None, 'mae': None, 'mape': None},
                'ranking': []
            },
            'test': {
                'learning': {'mse': None, 'r2': None, 'mae': None, 'mape': None},
                'ranking': []
            },
            'time': {
                'train': 0.0,
                'test': 0.0,
                'eval': 0.0
            }
        }
