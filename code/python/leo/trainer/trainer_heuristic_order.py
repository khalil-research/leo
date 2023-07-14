import logging

import numpy as np
import pandas as pd

from leo.utils.data import feat_names
from leo.utils.metrics import eval_learning_metrics, eval_order_metrics
from leo.utils.metrics import eval_rank_metrics
from leo.utils.order import get_variable_order
from leo.utils.order import get_variable_rank
from .trainer import Trainer

log = logging.getLogger(__name__)


class HeuristicOrderTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None, ps=None, rs=None):
        super(HeuristicOrderTrainer, self).__init__(data, model, cfg, ps, rs)

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
        y_train_rank, y_val_rank = get_variable_rank(scores=self.dtrain['y']), get_variable_rank(scores=self.dval['y'])
        self.rs['train']['learning'] = eval_learning_metrics(y_train_rank,
                                                             self.ps['train']['rank'],
                                                             sample_weight=self.dtrain['sample_weight'])
        log.info("** Validation learning metrics:")
        self.rs['val']['learning'] = eval_learning_metrics(y_val_rank,
                                                           self.ps['val']['rank'],
                                                           sample_weight=self.dval['sample_weight'])

        # Eval rank predictions
        log.info("** Train order metrics:")
        y_order = get_variable_order(scores=self.dtrain['y'])
        self.rs['train']['ranking'].extend(eval_order_metrics(y_order,
                                                              self.ps['train']['order'],
                                                              self.ps['train']['n_items']))
        self.rs['train']['ranking'].extend(eval_rank_metrics(y_train_rank,
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
        self.rs['val']['ranking'].extend(eval_rank_metrics(y_val_rank,
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

    def _get_split_rank(self, split):
        if self.cfg.model.name == 'HeuristicWeight':
            index = feat_names['vrank'].index('rk_{}_weight'.format(self.cfg.model.sort))
            ranks = [inst['x']['vrank'][:, index]
                     for inst in self.data[split]]

        elif self.cfg.model.name == 'HeuristicValue':
            index = feat_names['vrank'].index('rk_{}_value.{}'.format(self.cfg.model.sort, self.cfg.model.agg))

            ranks = [inst['x']['vrank'][:, index]
                     for inst in self.data[split]]

        elif self.cfg.model.name == 'HeuristicValueByWeight':
            assert self.cfg.model.sort != 'asc'
            if self.cfg.model.agg != 'min':
                index = feat_names['vrank'].index('rk_{}_value.{}/wt'.format(self.cfg.model.sort, self.cfg.model.agg))
                ranks = [inst['x']['vrank'][:, index]
                         for inst in self.data[split]]
            else:
                index = feat_names['var'].index('value.min/wt')
                scores = [inst['x']['var'][:, index] for inst in self.data[split]]
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

        scores = [inst['x']['var'][:, index] for inst in self.data[split]]
        reverse = False if self.cfg.model.sort == 'asc' else True

        return get_variable_order(scores=scores, reverse=reverse)
