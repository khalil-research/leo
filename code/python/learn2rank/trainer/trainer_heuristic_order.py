from pathlib import Path

from .trainer import Trainer
import numpy as np
from learn2rank.utils.data import feat_names
from learn2rank.utils.order import get_variable_rank
from learn2rank.utils.order import get_variable_order
import logging

from ..utils import eval_learning_metrics, eval_order_metrics
from ..utils.metrics import eval_rank_metrics
import pandas as pd

log = logging.getLogger(__name__)


class HeuristicOrderTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None, ps=None, rs=None):
        super(HeuristicOrderTrainer, self).__init__(data, model, cfg, ps, rs)

        self.res_path = Path(self.cfg.res_path[self.cfg.machine])
        self.dtrain = self._get_split_data(split='train')
        self.dval = self._get_split_data(split='val')
        self.dtest = self._get_split_data(split='test')

        if self.rs is None:
            self.rs = self._get_results_store()
            self.rs['task'] = 'point_regress'
            self.rs['model_name'] = self.cfg.model.name

        if self.ps is None:
            self.ps = self._get_preds_store()

            self.ps['train']['names'] = self.dtrain['names']
            self.ps['train']['n_items'] = self.dtrain['n_items']

            self.ps['val']['names'] = self.dval['names']
            self.ps['val']['n_items'] = self.dval['n_items']

            self.ps['test']['names'] = self.dtest['names']
            self.ps['test']['n_items'] = self.dtest['n_items']

    def run(self):
        self.ps['train']['rank'] = self._get_split_rank(split='train')
        self.ps['val']['rank'] = self._get_split_rank(split='val')

        self.ps['train']['order'] = self._get_split_order(split='train')
        self.ps['val']['order'] = self._get_split_order(split='val')

        # Eval learning metrics
        log.info(f"* {self.cfg.model.name} Results")
        log.info("** Train learning metrics:")
        self.rs['train']['learning'] = eval_learning_metrics(get_variable_rank(scores=self.dtrain['y']),
                                                             self.ps['train']['score'],
                                                             sample_weight=self.dtrain['wt'])
        log.info("** Validation learning metrics:")
        self.rs['val']['learning'] = eval_learning_metrics(get_variable_rank(scores=self.dval['y']),
                                                           self.ps['val']['score'],
                                                           sample_weight=self.dval['wt'])

        # Eval rank predictions
        log.info("** Train order metrics:")
        y_order = get_variable_order(scores=self.dtrain['y'])
        self.rs['train']['ranking'].extend(eval_order_metrics(y_order,
                                                              self.ps['train']['order'],
                                                              self.ps['train']['n_items']))
        self.rs['train']['ranking'].extend(eval_rank_metrics(y_order,
                                                             self.ps['train']['rank'],
                                                             self.ps['train']['n_items']))
        df_train = pd.DataFrame(self.rs['train']['ranking'],
                                columns=['id', 'metric_type', 'metric_value'])
        self.print_rank_metrics(df_train)

        log.info("** Val order metrics:")
        y_order = get_variable_order(scores=self.dval['y'])
        self.rs['val']['ranking'].extend(eval_order_metrics(y_order,
                                                            self.ps['val']['order'],
                                                            self.ps['val']['n_items']))
        self.rs['val']['ranking'].extend(eval_rank_metrics(y_order,
                                                           self.ps['val']['rank'],
                                                           self.ps['val']['n_items']))
        df_val = pd.DataFrame(self.rs['val']['ranking'],
                              columns=['id', 'metric_type', 'metric_value'])
        self.print_rank_metrics(df_val)

        self.val_tau = df_val.query("metric_type == 'kendall-coeff'")['metric_value'].mean()

        self._save_predictions()
        self._save_results()

    def predict(self, split='test'):
        self.ps[split]['rank'] = self._get_split_rank(split=split)
        self.ps[split]['order'] = self._get_split_order(split=split)

        self._save_predictions()
        self._save_results()

    def _get_split_data(self, split='train'):
        x, y, wt, names, n_items = [], [], [], [], []

        size = self.cfg.problem.size
        # for size in self.cfg.dataset.size:
        for v in self.data[size][split]:
            _x, _y = v['x'], v['y']
            x.append(_x)
            y.append(_y)
            n_items.append(self.cfg.problem.n_vars)
            names.append(v['name'])

        sample_weights = [1] * len(y)

        return {'x': x, 'y': y, 'names': names, 'n_items': n_items, 'wt': sample_weights}

    def _get_split_rank(self, split):
        if self.cfg.model.name == 'HeuristicWeight':
            index = feat_names['vrank'].index('rk_{}_weight'.format(self.cfg.model.sort))
            ranks = [inst['x']['vrank'][:, index]
                     for inst in self.data[self.cfg.problem.size][split]]

        elif self.cfg.model.name == 'HeuristicValue':
            index = feat_names['vrank'].index('rk_{}_value.{}'.format(self.cfg.model.sort, self.cfg.model.agg))

            ranks = [inst['x']['vrank'][:, index]
                     for inst in self.data[self.cfg.problem.size][split]]

        elif self.cfg.model.name == 'HeuristicValueByWeight':
            assert self.cfg.model.sort != 'asc'
            if self.cfg.model.agg != 'min':
                index = feat_names['vrank'].index('rk_{}_value.{}/wt'.format(self.cfg.model.sort, self.cfg.model.agg))
                ranks = [inst['x']['vrank'][:, index]
                         for inst in self.data[self.cfg.problem.size][split]]
            else:
                index = feat_names['var'].index('value.min/wt')
                scores = [inst['x']['var'][:, index] for inst in self.data[self.cfg.problem.size][split]]
                ranks = get_variable_rank(scores=scores, reverse=True)

        else:
            raise ValueError('Invalid model name!')

        if np.max(ranks[0]) <= 1:
            ranks = np.array(ranks) * self.cfg.problem.n_vars

        return ranks

    def _get_split_order(self, split):
        if self.cfg.model.name == 'HeuristicWeight':
            index = feat_names['var'].index('weight')

        elif self.cfg.model.name == 'HeuristicValue':
            index = feat_names['var'].index('value.{}'.format(self.cfg.model.agg))

        elif self.cfg.model.name == 'HeuristicValueByWeight':
            assert self.cfg.model.sort != 'asc'
            index = feat_names['var'].index('value.{}/wt'.format(self.cfg.model.agg))

        else:
            raise ValueError('Invalid model name!')

        scores = [inst['x']['var'][:, index] for inst in self.data[self.cfg.problem.size][split]]
        reverse = False if self.cfg.model.sort == 'asc' else True

        return get_variable_order(scores=scores, reverse=reverse)
