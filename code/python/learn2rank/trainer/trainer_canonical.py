from pathlib import Path

from .trainer import Trainer


class CanonicalTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None, ps=None, rs=None):
        super(CanonicalTrainer, self).__init__(data, model, cfg)

        self.res_path = Path(self.cfg.res_path[self.cfg.machine])

        self.rs = rs
        self.rs = self._get_results_store() if self.rs is None else self.rs
        self.rs['task'] = self.cfg.task
        self.rs['model_name'] = self.cfg.model.name
        if 'test' not in self.rs:
            self.rs['test'] = {'learning': {}, 'ranking': []}

        self.ps = ps
        self.ps = self._get_preds_store() if self.ps is None else self.ps
        if 'test' not in self.ps:
            self.ps['test'] = {'names': [], 'n_items': [], 'score': [], 'rank': [], 'order': []}

        names_tr, n_items_tr, wt_tr = self._get_split_data(split='train')
        self.ps['tr']['names'] = names_tr
        self.ps['tr']['n_items'] = n_items_tr

        names_val, n_items_val, wt_val = self._get_split_data(split='val')
        self.ps['val']['names'] = names_val
        self.ps['val']['n_items'] = n_items_val

        names_test, n_items_test, wt_test = self._get_split_data(split='test')
        self.ps['test']['names'] = names_test
        self.ps['test']['n_items'] = n_items_test

    def run(self):
        self.ps['tr']['order'] = [list(range(self.cfg.problem.n_vars)) for _ in self.ps['tr']['names']]
        self.ps['val']['order'] = [list(range(self.cfg.problem.n_vars)) for _ in self.ps['val']['names']]

        self._save_predictions()
        self._save_results()

    def predict(self, split='test'):
        split = 'tr' if split == 'train' else split

        self.ps[split]['order'] = [list(range(self.cfg.problem.n_vars)) for _ in self.ps[split]['names']]
        self._save_predictions()
        self._save_results()

    def _get_split_data(self, split='train'):
        x, y, wt, names, n_items = [], [], [], [], []

        size = self.cfg.problem.size
        # for size in self.cfg.dataset.size:
        for v in self.data[size][split]:
            _y = v['y']
            n_items.append(self.cfg.problem.n_vars)
            names.append(v['name'])
            y.append(_y)

        sample_weights = [1] * len(y)

        return names, n_items, sample_weights
