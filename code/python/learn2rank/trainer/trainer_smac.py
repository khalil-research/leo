import ast
from pathlib import Path

import pandas as pd

from learn2rank.utils.data import read_data_from_file
from learn2rank.utils.order import get_variable_order_from_weights
from .trainer import Trainer


class SmacOneTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None):
        super(SmacOneTrainer, self).__init__(data, model, cfg)

        self.res_path = Path(self.cfg.res_path[self.cfg.machine])
        self.inst_root_path = self.res_path / 'instances' / cfg.problem.name
        self.label_path = self.res_path / 'labels' / cfg.problem.name / cfg.problem.size
        self.label_path = self.label_path / f'label_{cfg.problem.size}.csv'
        self.rs = self._get_results_store()
        self.ps = self._get_preds_store()
        self.rs['task'] = self.cfg.task
        self.rs['model_name'] = self.cfg.model.name

        self.min_weight_incb = {'avg_value': 0.0,
                                'avg_value_by_weight': 0.0,
                                'max_value': 0.0,
                                'max_value_by_weight': 0.0,
                                'min_value': 0.0,
                                'min_value_by_weight': 0.0,
                                'weight': -1.0}

    def run(self):
        names_tr, n_items_tr, wt_tr = self._get_split_data(split='train')
        self.ps['tr']['names'] = names_tr
        self.ps['tr']['n_items'] = n_items_tr

        names_val, n_items_val, wt_val = self._get_split_data(split='val')
        self.ps['val']['names'] = names_val
        self.ps['val']['n_items'] = n_items_val

        self.ps['tr']['order'] = self.predict(names_tr, 'train')
        self.ps['val']['order'] = self.predict(names_val, 'val')

        self._save_predictions()
        self._save_results()

    def predict(self, names, split):
        y_pred_lst = []
        df = pd.read_csv(self.label_path)
        for name in names:
            acronym, _, a, b, pid = name.split("_")
            if acronym == 'kp':
                size = f'{a}_{b}'

                inst = self.inst_root_path / size / split / f'{name}.dat'
                data = read_data_from_file(acronym, inst)

                df1 = df[df['pid'] == pid]
                if df1.shape[0]:
                    incb = ast.literal_eval(df1.iloc[0]['incb'])
                else:
                    incb = self.min_weight_incb
                y_pred, _ = get_variable_order_from_weights(data, incb)
                y_pred_lst.append(y_pred)

        return y_pred_lst

    def _get_split_data(self, split='train'):
        x, y, wt, names, n_items = [], [], [], [], []

        size = self.cfg.problem.size
        # for size in self.cfg.dataset.size:
        for v in self.data[size][split]:
            _y = v['y']
            n_items.append(len(_y))
            names.append(v['name'])
            y.append(_y)

        sample_weights = [1] * len(y)

        return names, n_items, sample_weights

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
            },
            'test': {
                'names': [],
                'n_items': [],
                'score': [],
                'rank': [],
                'order': []
            }
        }
