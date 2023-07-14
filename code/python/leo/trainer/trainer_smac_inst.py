import ast

import pandas as pd

from leo import path
from leo.utils.data import read_data_from_file
from leo.utils.order import get_variable_order
from leo.utils.order import get_variable_rank
from leo.utils.order import get_variable_score_from_weights
from .trainer import Trainer


class SmacITrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None, ps=None, rs=None):
        super(SmacITrainer, self).__init__(data, model, cfg, ps, rs)

        self.inst_root_path = path.instances / cfg.problem.name
        self.label_path = path.label / cfg.problem.name / cfg.problem.size
        self.label_path = self.label_path / f'label_{cfg.problem.size}.csv'
        self.min_weight_incb = {'avg_value': 0.0,
                                'avg_value_by_weight': 0.0,
                                'max_value': 0.0,
                                'max_value_by_weight': 0.0,
                                'min_value': 0.0,
                                'min_value_by_weight': 0.0,
                                'weight': -1.0}
        if self.rs is None:
            self.rs = self._get_results_store()
            self.rs['task'] = self.cfg.task
            self.rs['model_name'] = self.cfg.model.name

        if self.ps is None:
            self.ps = self._get_preds_store()

            names_tr, n_items_tr, wt_tr = self._get_split_data(split='train')
            self.ps['train']['names'] = names_tr
            self.ps['train']['n_items'] = n_items_tr

            names_val, n_items_val, wt_val = self._get_split_data(split='val')
            self.ps['val']['names'] = names_val
            self.ps['val']['n_items'] = n_items_val

            names_test, n_items_test, wt_test = self._get_split_data(split='test')
            self.ps['test']['names'] = names_test
            self.ps['test']['n_items'] = n_items_test

    def run(self):
        self.ps['train']['score'] = self._get_split_scores(split='train')
        self.ps['val']['score'] = self._get_split_scores(split='val')

        self.ps['train']['order'] = get_variable_order(scores=self.ps['train']['score'], reverse=True)
        self.ps['val']['order'] = get_variable_order(scores=self.ps['val']['score'], reverse=True)

        self.ps['train']['rank'] = get_variable_rank(scores=self.ps['train']['score'], reverse=True)
        self.ps['val']['rank'] = get_variable_rank(scores=self.ps['val']['score'], reverse=True)

        self.rs['val']['ranking'] = [[0, 'kendall-coeff', 1]]
        self.val_tau = 1

        self._save_predictions()
        self._save_results()

    def predict(self, *args, **kwargs):
        pass

    def _get_split_data(self, split='train'):
        x, y, wt, names, n_items = [], [], [], [], []

        # for size in self.cfg.dataset.size:
        for v in self.data[split]:
            _y = v['y']
            n_items.append(len(_y))
            names.append(v['name'])
            y.append(_y)

        sample_weights = [1] * len(y)

        return names, n_items, sample_weights

    def _get_split_scores(self, split='train'):
        scores = []
        df = pd.read_csv(self.label_path)
        for name in self.ps[split]['names']:
            acronym, _, a, b, pid = name.split("_")
            if acronym == 'kp':
                size = f'{a}_{b}'

                inst = self.inst_root_path / size / split / f'{name}.dat'
                data = read_data_from_file(acronym, inst)

                df1 = df.query(f"pid == '{pid}'")
                if df1.shape[0]:
                    incb = ast.literal_eval(df1.iloc[0]['incb'])
                else:
                    incb = self.min_weight_incb

                scores.append(get_variable_score_from_weights(data, incb))

        return scores
