import logging
import pickle
import time

import numpy as np
import pandas as pd

from leo.utils.data import get_sample_weight
from leo.utils.metrics import eval_learning_metrics
from leo.utils.metrics import eval_order_metrics
from leo.utils.metrics import eval_rank_metrics
from leo.utils.order import get_variable_order
from leo.utils.order import get_variable_rank
from .trainer import Trainer

log = logging.getLogger(__name__)


class SklearnTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None, ps=None, rs=None):
        super().__init__(data, model, cfg, ps, rs)

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
        # Train
        self.rs['time']['training'] = time.time()
        self.model.fit(self.dtrain['x'], self.dtrain['y'], sample_weight=self.dtrain['sample_weight'])
        self.rs['time']['training'] = time.time() - self.rs['time']['training']

        # Predict scores
        # (n_samples x n_vars) x 1
        # Predict on train set
        self.rs['time']['prediction']['train'] = time.time()
        self.ps['train']['score'] = self.model.predict(self.dtrain['x'])
        self.rs['time']['prediction']['train'] = time.time() - self.rs['time']['prediction']['train']
        # Predict on val set
        self.rs['time']['prediction']['val'] = time.time()
        self.ps['val']['score'] = self.model.predict(self.dval['x'])
        self.rs['time']['prediction']['val'] = time.time() - self.rs['time']['prediction']['val']

        # Eval learning metrics
        log.info(f"* {self.cfg.model.name} Results")
        log.info("** Train learning metrics:")
        self.rs['train']['learning'] = eval_learning_metrics(self.dtrain['y'],
                                                             self.ps['train']['score'],
                                                             sample_weight=self.dtrain['sample_weight'])
        log.info("** Validation learning metrics:")
        self.rs['val']['learning'] = eval_learning_metrics(self.dval['y'],
                                                           self.ps['val']['score'],
                                                           sample_weight=self.dval['sample_weight'])

        # Unflatten data
        unflattened = list(map(self.unflatten_data,
                               (self.dtrain['y'],
                                self.ps['train']['score'],
                                self.dval['y'],
                                self.ps['val']['score']),
                               (self.ps['train']['n_items'],
                                self.ps['train']['n_items'],
                                self.ps['val']['n_items'],
                                self.ps['val']['n_items'])))
        self.dtrain['y'], self.ps['train']['score'] = unflattened[0], unflattened[1]
        self.dval['y'], self.ps['val']['score'] = unflattened[2], unflattened[3]

        # Get order
        self.ps['train']['order'] = get_variable_order(scores=self.ps['train']['score'])
        self.ps['val']['order'] = get_variable_order(scores=self.ps['val']['score'])
        # Get rank
        self.ps['train']['rank'] = get_variable_rank(scores=self.ps['train']['score'])
        self.ps['val']['rank'] = get_variable_rank(scores=self.ps['val']['score'])

        # Eval rank predictions
        log.info("** Train order metrics:")
        self.rs['train']['ranking'].extend(eval_order_metrics(get_variable_order(scores=self.dtrain['y']),
                                                              self.ps['train']['order'],
                                                              self.ps['train']['n_items']))
        self.rs['train']['ranking'].extend(eval_rank_metrics(get_variable_rank(scores=self.dtrain['y']),
                                                             self.ps['train']['rank'],
                                                             self.ps['train']['n_items']))
        df_train = pd.DataFrame(self.rs['train']['ranking'],
                                columns=['id', 'metric_type', 'metric_value'])
        self.print_rank_metrics(df_train)

        log.info("** Val order metrics:")
        self.rs['val']['ranking'].extend(eval_order_metrics(get_variable_order(scores=self.dval['y']),
                                                            self.ps['val']['order'],
                                                            self.ps['val']['n_items']))
        self.rs['val']['ranking'].extend(eval_rank_metrics(get_variable_rank(scores=self.dval['y']),
                                                           self.ps['val']['rank'],
                                                           self.ps['val']['n_items']))
        df_val = pd.DataFrame(self.rs['val']['ranking'],
                              columns=['id', 'metric_type', 'metric_value'])
        self.print_rank_metrics(df_val)
        log.info(f"  {self.cfg.model.name} train time: {self.rs['time']['training']} \n")

        self.val_tau = df_val.query("metric_type == 'kendall-coeff'")['metric_value'].mean()
        self._save_model()
        self._save_predictions()
        self._save_results()

    def predict(self, split='test'):
        dsplit = getattr(self, f'd{split}')
        self.rs['time']['prediction'][split] = time.time()
        self.ps[split]['score'] = self.model.predict(dsplit['x'])
        self.rs['time']['prediction'][split] = time.time() - self.rs['time']['prediction'][split]

        self.ps[split]['score'] = self.unflatten_data(self.ps[split]['score'], self.ps[split]['n_items'])

        self.ps[split]['order'] = get_variable_order(scores=self.ps[split]['score'])
        self.ps[split]['rank'] = get_variable_rank(scores=self.ps[split]['score'])

        self._save_predictions()
        self._save_results()

    def _save_model(self):
        model_path_root = self._get_path('pretrained')
        model_path_root.mkdir(parents=True, exist_ok=True)
        model_path = model_path_root / f'model_{self.model.id}.pkl'
        with open(model_path, 'wb') as p:
            pickle.dump(self.model, p)

    def _get_split_data(self, split='train'):
        x, y, wt, names, n_items = [], [], [], [], []
        size = self.cfg.problem.size
        # for size in self.cfg.problem.size:
        n_objs, n_vars = list(map(int, size.split('_')))
        for v in self.data[split]:
            _x = v['x']
            n_items.append(n_vars)
            names.append(v['name'])
            x.extend(np.hstack((_x['var'], _x['vrank'], _x['inst'])))

            if split != 'test':
                _y = v['y']
                weights = get_sample_weight(_y, bool(self.cfg.model.weights))
                y.extend(_y)
                wt.extend(list(weights[0]))

        return {'x': np.asarray(x), 'y': np.asarray(y), 'names': names, 'n_items': n_items,
                'sample_weight': np.asarray(wt)}
