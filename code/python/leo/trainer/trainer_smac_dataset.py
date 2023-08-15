import ast
import logging

import pandas as pd

from leo import path
from leo.utils.data import read_data_from_file
from leo.utils.metrics import eval_learning_metrics
from leo.utils.metrics import eval_order_metrics
from leo.utils.metrics import eval_rank_metrics
from leo.utils.order import get_variable_order
from leo.utils.order import get_variable_rank
from leo.utils.order import get_variable_score_from_weights
from .trainer import Trainer

log = logging.getLogger(__name__)


class SmacDTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None, rs=None, ps=None):
        super(SmacDTrainer, self).__init__(data, model, cfg, ps, rs)

        self.inst_root_path = path.instances / cfg.problem.name
        self.traj_path = path.SmacD / cfg.problem.name / cfg.problem.size
        # TODO: Create label file to read SmacD incumbent instead of hard coding the run path
        self.traj_path = self.traj_path / f'{cfg.problem.acronym}_7_{cfg.problem.size}_0' / 'run_777' / 'traj.json'
        self.incb = ast.literal_eval(self.traj_path.read_text().strip().split('\n')[-1])["incumbent"]

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
        self.ps['train']['score'] = self._get_split_scores(split='train')
        self.ps['val']['score'] = self._get_split_scores(split='val')

        self.ps['train']['order'] = get_variable_order(scores=self.ps['train']['score'], reverse=True)
        self.ps['val']['order'] = get_variable_order(scores=self.ps['val']['score'], reverse=True)

        self.ps['train']['rank'] = get_variable_rank(scores=self.ps['train']['score'], reverse=True)
        self.ps['val']['rank'] = get_variable_rank(scores=self.ps['val']['score'], reverse=True)

        # Eval learning metrics
        log.info(f"* {self.cfg.model.name} Results")
        log.info("** Train learning metrics:")
        self.rs['train']['learning'] = eval_learning_metrics(self.dtrain['y'],
                                                             self.ps['train']['rank'],
                                                             sample_weight=self.dtrain['sample_weight'])
        log.info("** Validation learning metrics:")
        self.rs['val']['learning'] = eval_learning_metrics(self.dval['y'],
                                                           self.ps['val']['rank'],
                                                           sample_weight=self.dval['sample_weight'])

        # Eval rank predictions
        log.info("** Train order metrics:")
        self.rs['train']['ranking'].extend(eval_order_metrics(get_variable_order(scores=self.dtrain['y']),
                                                              self.ps['train']['order'],
                                                              self.ps['train']['n_items']))
        self.rs['train']['ranking'].extend(eval_rank_metrics(self.dtrain['y'],
                                                             self.ps['train']['rank'],
                                                             self.ps['train']['n_items']))
        df_train = pd.DataFrame(self.rs['train']['ranking'], columns=['id', 'metric_type', 'metric_value'])
        self.print_rank_metrics(df_train)

        log.info("** Val order metrics:")
        self.rs['val']['ranking'].extend(eval_order_metrics(get_variable_order(scores=self.dval['y']),
                                                            self.ps['val']['order'],
                                                            self.ps['val']['n_items']))
        self.rs['val']['ranking'].extend(eval_rank_metrics(self.dval['y'],
                                                           self.ps['val']['rank'],
                                                           self.ps['val']['n_items']))
        df_val = pd.DataFrame(self.rs['val']['ranking'], columns=['id', 'metric_type', 'metric_value'])
        self.print_rank_metrics(df_val)
        log.info(f"  {self.cfg.model.name} \n")
        self.val_tau = df_val.query("metric_type == 'kendall-coeff'")['metric_value'].mean()

        self._save_predictions()
        self._save_results()

    def predict(self, split='test'):
        self.ps[split]['score'] = self._get_split_scores(split=split)

        self.ps[split]['order'] = get_variable_order(scores=self.ps[split]['score'], reverse=True)
        self.ps[split]['rank'] = get_variable_rank(scores=self.ps[split]['score'], reverse=True)

        self._save_predictions()
        self._save_results()

    def _get_split_scores(self, split='train'):
        scores = []

        for name in self.ps[split]['names']:
            acronym, _, a, b, pid = name.split("_")
            if acronym == 'kp':
                size = f'{a}_{b}'

                inst = self.inst_root_path / size / split / f'{name}.dat'
                data = read_data_from_file(acronym, inst)
                scores.append(get_variable_score_from_weights(data, self.incb))

        return scores
