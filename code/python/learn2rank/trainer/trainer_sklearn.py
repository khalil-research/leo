import logging
import pickle
import time

import numpy as np

# from learn2rank.utils.data import flatten_data, unflatten_data
# from learn2rank.utils.data import get_n_items, get_sample_weight
from learn2rank.utils.data import get_sample_weight, unflatten_data
from learn2rank.utils.metrics import eval_learning_metrics
from learn2rank.utils.metrics import eval_order_metrics
from learn2rank.utils.metrics import eval_rank_metrics
from learn2rank.utils.order import pred_score2order
from learn2rank.utils.order import pred_score2rank
from .trainer import Trainer

log = logging.getLogger(__name__)


class SklearnTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None):
        super().__init__(data, model, cfg)

        self.rs = self._get_results_store()
        self.rs['task'] = self.cfg.task
        self.rs['model_name'] = self.cfg.model.name
        self.ps = self._get_preds_store()

    def run(self):
        x_tr, y_tr, names_tr, n_items_tr, wt_tr = self._get_split_data(split='train')
        x_val, y_val, names_val, n_items_val, wt_val = self._get_split_data(split='val')
        self.ps['tr']['names'], self.ps['tr']['n_items'] = names_tr, n_items_tr
        self.ps['val']['names'], self.ps['val']['n_items'] = names_val, n_items_val
        # self.ps['tr']['n_items'] = get_n_items(y_tr)
        # self.ps['val']['n_items'] = get_n_items(y_val)
        # wt_tr = get_sample_weight(y_tr, self.cfg.model.weights)
        # wt_val = get_sample_weight(y_val, self.cfg.model.weights)

        # _data = flatten_data([x_tr, y_tr, wt_tr, x_val, y_val, wt_val])
        # [x_tr, y_tr, wt_tr, x_val, y_val, wt_val] = _data

        # Train
        _time = time.time()
        self.model.fit(x_tr, y_tr, sample_weight=wt_tr)
        _time = time.time() - _time
        self.rs['time']['train'] = _time

        # Predict
        self.ps['tr']['score'] = self.model.predict(x_tr)
        self.ps['val']['score'] = self.model.predict(x_val)

        # Eval learning metrics
        log.info(f"* {self.cfg.model.name} Results")
        log.info("** Train learning metrics:")
        self.rs['tr']['learning'] = eval_learning_metrics(y_tr, self.ps['tr']['score'], sample_weight=wt_tr)
        log.info("** Validation learning metrics:")
        self.rs['val']['learning'] = eval_learning_metrics(y_val, self.ps['val']['score'], sample_weight=wt_val)

        # Unflatten data
        y_tr, self.ps['tr']['score'] = unflatten_data([y_tr, self.ps['tr']['score']],
                                                      n_items=self.ps['tr']['n_items'])
        y_val, self.ps['val']['score'] = unflatten_data([y_val, self.ps['val']['score']],
                                                        n_items=self.ps['val']['n_items'])

        # Transform scores to ranks
        self.ps['tr']['rank'] = pred_score2rank(self.ps['tr']['score'])
        self.ps['val']['rank'] = pred_score2rank(self.ps['val']['score'])

        # Transform scores to order
        y_tr_order = pred_score2order(y_tr)
        self.ps['tr']['order'] = pred_score2order(self.ps['tr']['score'])
        y_val_order = pred_score2order(y_val)
        self.ps['val']['order'] = pred_score2order(self.ps['val']['score'])

        # Eval rank predictions
        log.info("** Train order metrics:")
        self.rs['tr']['ranking'].extend(eval_order_metrics(y_tr_order,
                                                           self.ps['tr']['order'],
                                                           self.ps['tr']['n_items']))
        self.rs['tr']['ranking'].extend(eval_rank_metrics(y_tr, self.ps['tr']['rank'], self.ps['tr']['n_items']))

        log.info("** Val order metrics:")
        self.rs['val']['ranking'].extend(eval_order_metrics(y_val_order,
                                                            self.ps['val']['order'],
                                                            self.ps['val']['n_items']))
        self.rs['val']['ranking'].extend(eval_rank_metrics(y_val, self.ps['val']['rank'], self.ps['val']['n_items']))

        log.info(f"  {self.cfg.model.name} train time: {self.rs['time']['train']} \n")

        self._save_model()
        self._save_predictions()
        self._save_results()

    def predict(self, x=None):
        assert x is not None

        return self.model.predict(x)

    def _save_model(self):
        model_path = self.res_path / f'pretrained/{self.cfg.problem.name}/{self.cfg.problem.size}'
        model_path.mkdir(parents=True, exist_ok=True)
        model_path = model_path / f'model_{self.model.id}.pkl'
        with open(model_path, 'wb') as p:
            pickle.dump(self.ps, p)

    def _get_split_data(self, split='train'):
        x, y, wt, names, n_items = [], [], [], [], []
        size = self.cfg.problem.size
        # for size in self.cfg.problem.size:
        for v in self.data[size][split]:
            _x, _y = v['x'], v['y']
            n_items.append(len(_y))
            names.append(v['name'])

            x.extend(np.hstack((_x['var'], _x['vrank'], _x['inst'])))

            weights = get_sample_weight(_y, bool(self.cfg.model.weights))
            y.extend(_y)
            wt.extend(list(weights[0]))

        return np.asarray(x), np.asarray(y), names, n_items, np.asarray(wt)

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
