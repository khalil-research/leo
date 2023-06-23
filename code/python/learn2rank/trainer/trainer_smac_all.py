import ast
from pathlib import Path

from learn2rank.utils.data import read_data_from_file
from learn2rank.utils.order import get_variable_order_from_weights
from .trainer import Trainer


class SmacAllTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None, rs=None, ps=None):
        super(SmacAllTrainer, self).__init__(data, model, cfg, ps, rs)

        self.res_path = Path(self.cfg.res_path[self.cfg.machine])
        self.inst_root_path = self.res_path / 'instances' / cfg.problem.name
        self.traj_path = self.res_path / 'smac_all_output' / cfg.problem.name / cfg.problem.size
        self.traj_path = self.traj_path / f'{cfg.problem.acronym}_7_{cfg.problem.size}_0' / 'run_777' / 'traj.json'
        self.incb = ast.literal_eval(self.traj_path.read_text().strip().split('\n')[-1])["incumbent"]

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
        self.ps['train']['order'] = self._get_split_order(split='train')
        self.ps['val']['order'] = self._get_split_order(split='val')
        self.val_tau = -1

        self._save_predictions()
        self._save_results()

    def predict(self, split='test'):
        self.ps[split]['order'] = self._get_split_order(split=split)

        self._save_predictions()
        self._save_results()

    def _get_split_order(self, split='train'):
        y_pred_lst = []

        for name in self.ps[split]['names']:
            acronym, _, a, b, pid = name.split("_")
            if acronym == 'kp':
                size = f'{a}_{b}'

                inst = self.inst_root_path / size / split / f'{name}.dat'
                data = read_data_from_file(acronym, inst)
                y_pred, _ = get_variable_order_from_weights(data, self.incb)
                y_pred_lst.append(y_pred)

        return y_pred_lst

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
