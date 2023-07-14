import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from leo import path

log = logging.getLogger(__name__)


class Trainer(ABC):
    def __init__(self, data, model, config, ps, rs):
        self.data = data
        self.model = model
        self.cfg = config
        self.ps = ps
        self.rs = rs

        self.pred_path_root = self._get_path('prediction')
        self.pred_path_root.mkdir(parents=True, exist_ok=True)

        self.val_tau = None
        if self.rs is not None and 'val' in self.rs and 'ranking' in self.rs['val']:
            df = pd.DataFrame(self.rs['val']['ranking'], columns=['id', 'name', 'value'])
            if df.shape[0]:
                self.val_tau = np.mean(df.query("name == 'kendall-coeff'")['value'])

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def _save_predictions(self):
        pred_path = self.pred_path_root / f'prediction_{self.model.id}.pkl'
        with open(pred_path, 'wb') as p:
            pickle.dump(self.ps, p)

    def _save_results(self):
        if self.val_tau is not None:
            Path(self.pred_path_root / f'val_tau_{self.model.id}.txt').write_text(str(self.val_tau))

        pred_path = self.pred_path_root / f'results_{self.model.id}.pkl'
        with open(pred_path, 'wb') as p:
            pickle.dump(self.rs, p)

    def _get_split_data(self, split='train'):
        x, y, wt, names, n_items = [], [], [], [], []

        # for size in self.cfg.dataset.size:
        for v in self.data[split]:
            _x, _y = v['x'], v['y']
            x.append(_x)
            y.append(_y)
            n_items.append(self.cfg.problem.n_vars)
            names.append(v['name'])
        sample_weights = [1] * len(y)

        return {'x': x, 'y': y, 'names': names, 'n_items': n_items, 'sample_weight': sample_weights}

    def _get_preds_store(self):
        return {
            'task': self.cfg.task,
            'model_name': self.cfg.model.name,
            'model_id': self.model.id,
            'model_params': str(self.model),
            'train': {
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
            },
            'test': {
                'names': [],
                'n_items': [],
                'score': [],
                'rank': [],
                'order': []
            }
        }

    def _get_results_store(self):
        return {
            'task': self.cfg.task,
            'model_name': self.cfg.model.name,
            'model_id': self.model.id,
            'model_params': str(self.model),
            'train': {
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

    def _get_path(self, kind):
        if self.cfg.context and self.cfg.fused:
            return path[kind] / self.cfg.problem.name / 'all_context'
        elif not self.cfg.context and self.cfg.fused:
            return path[kind] / self.cfg.problem.name / 'all'
        elif self.cfg.context and not self.cfg.fused:
            return path[kind] / self.cfg.problem.name / f'{self.cfg.problem.size}_context'
        else:
            return path[kind] / self.cfg.problem.name / self.cfg.problem.size

    @staticmethod
    def print_rank_metrics(df):
        log.info(f"Top 10 Common  : {df[df['metric_type'] == 'top_10_common']['metric_value'].mean()} +/- "
                 f"{df[df['metric_type'] == 'top_10_common']['metric_value'].std()} ")
        log.info(
            f"Top 10 Same : {df[df['metric_type'] == 'top_10_same']['metric_value'].mean()} +/- "
            f"{df[df['metric_type'] == 'top_10_same']['metric_value'].std()} ")
        log.info(
            f"Top 10 Penalty    : {df[df['metric_type'] == 'top_10_penalty']['metric_value'].mean()} +/- "
            f"{df[df['metric_type'] == 'top_10_penalty']['metric_value'].std()} ")
        log.info(
            f"Top 5 Common   : {df[df['metric_type'] == 'top_5_common']['metric_value'].mean()} +/- "
            f"{df[df['metric_type'] == 'top_5_common']['metric_value'].std()} ")
        log.info(
            f"Top 5 Same  : {df[df['metric_type'] == 'top_5_same']['metric_value'].mean()} +/- "
            f"{df[df['metric_type'] == 'top_5_same']['metric_value'].std()} ")
        log.info(
            f"Top 5 Penalty     : {df[df['metric_type'] == 'top_5_penalty']['metric_value'].mean()} +/- "
            f"{df[df['metric_type'] == 'top_5_penalty']['metric_value'].std()} ")
        log.info(f"Spearman Correlation    : {df[df['metric_type'] == 'spearman-coeff']['metric_value'].mean()} +/- "
                 f"{df[df['metric_type'] == 'spearman-coeff']['metric_value'].std()}")
        log.info(f"Kendall Correlation     : {df[df['metric_type'] == 'kendall-coeff']['metric_value'].mean()} +/- "
                 f"{df[df['metric_type'] == 'kendall-coeff']['metric_value'].std()}")

    @staticmethod
    def unflatten_data(x, group):
        x_unflat = []

        i = 0
        for g in group:
            x_unflat.append(x[i: i + g])
            i += g

        return x_unflat
