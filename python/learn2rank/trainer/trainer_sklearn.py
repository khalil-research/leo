import logging
import pickle
import time

import numpy as np

# from learn2rank.utils.data import flatten_data, unflatten_data
# from learn2rank.utils.data import get_n_items, get_sample_weight
from learn2rank.utils.data import get_sample_weight, unflatten_data
from learn2rank.utils.metrics import eval_learning_metrics
from learn2rank.utils.metrics import eval_order_metrics
from learn2rank.utils.order import score2order
from .trainer import Trainer

log = logging.getLogger(__name__)


class SklearnTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None):
        super().__init__(data, model, cfg)

        self.rs = self._get_results_store()
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
        self.model.train(x_tr, y_tr, sample_weight=wt_tr)
        _time = time.time() - _time
        self.rs['time']['train'] = _time

        # Predict
        self.ps['tr']['score'] = self.model(x_tr)
        self.ps['val']['score'] = self.model(x_val)

        # Eval learning metrics
        log.info("* Linear Regression Results")
        log.info("** Train learning metrics:")
        self.rs['tr']['learning'] = eval_learning_metrics(y_tr, self.ps['tr']['score'], sample_weight=wt_tr)
        log.info("** Validation learning metrics:")
        self.rs['val']['learning'] = eval_learning_metrics(y_val, self.ps['val']['score'], sample_weight=wt_val)

        # Unflatten data
        y_tr, self.ps['tr']['score'] = unflatten_data([y_tr, self.ps['tr']['score']], self.ps['tr']['n_items'])
        y_val, self.ps['val']['score'] = unflatten_data([y_val, self.ps['val']['score']], self.ps['val']['n_items'])
        
        # Transform scores to order
        y_tr_order = score2order(y_tr)
        self.ps['tr']['order'] = score2order(self.ps['tr']['score'])
        y_val_order = score2order(y_val)
        self.ps['val']['order'] = score2order(self.ps['val']['score'])

        # Eval rank predictions
        log.info("** Train order metrics:")
        self.rs['tr']['ranking'] = eval_order_metrics(y_tr_order, self.ps['tr']['order'], self.ps['tr']['n_items'])
        log.info("** Val order metrics:")
        self.rs['val']['ranking'] = eval_order_metrics(y_val_order, self.ps['val']['order'], self.ps['val']['n_items'])

        log.info(f"  Linear regression train time: {self.rs['time']['train']} \n")

        self._save_model()
        self._save_predictions()
        self._save_results()

    def predict(self, split='test'):
        pass

    def _save_model(self):
        self.model.save()

    def _save_predictions(self):
        with open('./predictions.pkl', 'wb') as p:
            pickle.dump(self.ps, p)

    def _save_results(self):
        with open('./results.pkl', 'wb') as p:
            pickle.dump(self.rs, p)

    def _get_split_data(self, split='train'):
        x, y, wt, names, n_items = [], [], [], [], []

        for size in self.cfg.dataset.size:
            for name, v in self.data[size][split].items():
                _x, _y = v['x'], v['y'][-1]
                size_blobs = size.split('_')
                n_items.append(size_blobs[1] if 'kp' in name else size_blobs[0])
                names.append(name)

                x.extend(np.hstack((_x['var'], _x['vrank'], _x['inst'])))

                weights = get_sample_weight(_y['rank'], self.cfg.model.weights)
                y.extend(_y['rank'])
                wt.extend(list(weights[0]))

        return np.asarray(x), np.asarray(y), names, n_items, np.asarray(wt)

    @staticmethod
    def _get_results_store():
        return {
            'tr': {
                'learning': {},
                'ranking': {},
            },
            'val': {
                'learning': {},
                'ranking': {}
            },
            'test': {
                'learning': {},
                'ranking': {}
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
                'order': []
            },
            'val': {
                'names': [],
                'n_items': [],
                'score': [],
                'order': []
            }
        }
