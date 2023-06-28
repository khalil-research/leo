import logging
import pandas as pd

from pathlib import Path

from .trainer import Trainer
from learn2rank.utils.metrics import eval_learning_metrics
from learn2rank.utils.metrics import eval_rank_metrics
from learn2rank.utils.metrics import eval_order_metrics
from learn2rank.utils.order import get_variable_rank
from learn2rank.utils.order import get_variable_order

log = logging.getLogger(__name__)


class LexTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None, ps=None, rs=None):
        super(LexTrainer, self).__init__(data, model, cfg, ps, rs)

        self.res_path = Path(self.cfg.res_path[self.cfg.machine])
        self.dtrain = self._get_split_data(split='train')
        self.dval = self._get_split_data(split='val')
        self.dtest = self._get_split_data(split='test')

        if self.rs is None:
            self.rs = self._get_results_store()
            self.rs['task'] = self.cfg.task
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
        self.ps['train']['order'] = [list(range(self.cfg.problem.n_vars)) for _ in self.ps['train']['names']]
        self.ps['val']['order'] = [list(range(self.cfg.problem.n_vars)) for _ in self.ps['val']['names']]

        self.ps['train']['rank'] = [list(range(self.cfg.problem.n_vars)) for _ in self.ps['train']['names']]
        self.ps['val']['rank'] = [list(range(self.cfg.problem.n_vars)) for _ in self.ps['val']['names']]

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
        self.ps[split]['order'] = [list(range(self.cfg.problem.n_vars)) for _ in self.ps[split]['names']]
        self.ps[split]['rank'] = [list(range(self.cfg.problem.n_vars)) for _ in self.ps[split]['names']]

        self._save_predictions()
        self._save_results()
