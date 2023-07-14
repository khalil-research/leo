import logging
import time

import pandas as pd

from leo.utils.metrics import eval_order_metrics
from leo.utils.metrics import eval_rank_metrics
from leo.utils.order import get_variable_order
from leo.utils.order import get_variable_rank
from .trainer import Trainer

log = logging.getLogger(__name__)


class XGBoostTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None, ps=None, rs=None):
        super().__init__(data, model, cfg, ps, rs)

        self.x_train, self.y_train = self.data['train', 'dataset']
        self.x_val, self.y_val = self.data['val', 'dataset']
        # Load test if it exists
        if self.data['test', 'dataset'] is not None:
            self.x_test, _ = self.data['test', 'dataset']
        else:
            self.x_test = None

        # Load result store
        if self.rs is None:
            self.rs = self._get_results_store()
            self.rs['task'] = self.cfg.task
            self.rs['model_name'] = self.cfg.model.name

        # Load pred store
        if self.ps is None:
            self.ps = self._get_preds_store()
            # Names
            self.ps['train']['names'] = self.data['train', 'names']
            self.ps['val']['names'] = self.data['val', 'names']
            self.ps['test']['names'] = self.data['test', 'names']
            # Number of items
            self.ps['train']['n_items'] = self.data['train', 'n_items']
            self.ps['val']['n_items'] = self.data['val', 'n_items']
            self.ps['test']['n_items'] = self.data['test', 'n_items']

        # Unflatten data
        unflattend = list(map(self.unflatten_data,
                              (self.x_train, self.y_train, self.x_val, self.y_val),
                              (self.ps['train']['n_items'],
                               self.ps['train']['n_items'],
                               self.ps['val']['n_items'],
                               self.ps['val']['n_items'])))
        self.x_train_uf, self.y_train_uf = unflattend[0], unflattend[1]
        self.x_val_uf, self.y_val_uf = unflattend[2], unflattend[3]
        self.x_test_uf = [] if self.x_test is None else self.unflatten_data(self.x_test,
                                                                            self.ps['test']['n_items'])

    def run(self):
        self.rs['time']['train'] = time.time()
        # Train
        # feature_weights = self._get_feature_importance(self.cfg.model.feature_importance,
        #                                                self.x_train.shape[1])
        self.model.fit(
            self.x_train,
            self.y_train,
            group=self.ps['train']['n_items'],
            eval_set=[(self.x_val, self.y_val)],
            eval_group=[self.ps['val']['n_items']]
        )

        self.rs['time']['train'] = time.time() - self.rs['time']['train']
        log.info(f"  {self.cfg.model.name} train time: {self.rs['time']['train']} \n")

        # Train pred
        self.ps['train']['score'] = [self.model.predict(x) for x in self.x_train_uf]
        # Val pred
        self.rs['time']['val'] = time.time()
        self.ps['val']['score'] = [self.model.predict(x) for x in self.x_val_uf]
        self.rs['time']['val'] = time.time() - self.rs['time']['val']

        # Score to order
        train_order = get_variable_order(scores=self.y_train_uf, reverse=True)
        self.ps['train']['order'] = get_variable_order(scores=self.ps['train']['score'], reverse=True)
        val_order = get_variable_order(scores=self.y_val_uf, reverse=True)
        self.ps['val']['order'] = get_variable_order(scores=self.ps['val']['score'], reverse=True)

        # Score to rank
        train_rank = get_variable_rank(scores=self.y_train_uf, reverse=True, high_to_low=True)
        self.ps['train']['rank'] = get_variable_rank(scores=self.ps['train']['score'], reverse=True, high_to_low=True)
        val_rank = get_variable_rank(scores=self.y_val_uf, reverse=True, high_to_low=True)
        self.ps['val']['rank'] = get_variable_rank(scores=self.ps['val']['score'], reverse=True, high_to_low=True)

        # Eval rank predictions
        log.info('** Train metrics:')
        self.rs['train']['ranking'].extend(eval_order_metrics(train_order,
                                                              self.ps['train']['order'],
                                                              self.ps['train']['n_items']))
        self.rs['train']['ranking'].extend(eval_rank_metrics(train_rank,
                                                             self.ps['train']['rank'],
                                                             self.ps['train']['n_items']))
        df_train = pd.DataFrame(self.rs['train']['ranking'],
                                columns=['id', 'metric_type', 'metric_value'])
        self.print_rank_metrics(df_train)

        log.info('** Val metrics:')
        self.rs['val']['ranking'].extend(eval_order_metrics(val_order,
                                                            self.ps['val']['order'],
                                                            self.ps['val']['n_items']))
        self.rs['val']['ranking'].extend(eval_rank_metrics(val_rank,
                                                           self.ps['val']['rank'],
                                                           self.ps['val']['n_items']))

        df_val = pd.DataFrame(self.rs['val']['ranking'], columns=['id', 'metric_type', 'metric_value'])
        self.print_rank_metrics(df_train)
        self.val_tau = df_val.query("metric_type == 'kendall-coeff'")['metric_value'].mean()

        log.info('Saving...')
        self._save_model()
        self._save_predictions()
        self._save_results()

    def predict(self, split):
        if split == 'test':
            assert self.x_test is not None
        log.info('Predicting on the {} set...'.format(split))

        self.rs['time'][split] = time.time()
        for x in getattr(self, f'x_{split}_uf'):
            self.ps[split]['score'].append(self.model.predict(x))
        self.rs['time'][split] = time.time() - self.rs['time'][split]

        self.ps['test']['order'] = get_variable_order(scores=self.ps[split]['score'], reverse=True)
        self.ps['test']['rank'] = get_variable_rank(scores=self.ps['test']['score'], reverse=True, high_to_low=True)

        log.info('Saving...')
        self._save_predictions()
        self._save_results()

    def eval_learning_metrics(self, split="train"):
        pass

    def _save_model(self):
        model_path_root = self._get_path('pretrained')
        model_path_root.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_path_root / f'model_{self.model.id}.txt')

    @staticmethod
    def _get_feature_importance(get_weights, size):
        imp = None
        if get_weights:
            if size == 18:
                imp = [1] * 18
            elif size == 37:
                imp = [0.5] * 18
                imp.extend([1, 1])
                imp.extend([0.6] * 17)

        return imp
