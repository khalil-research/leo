import logging
import os
import time

import pandas as pd

from leo import path
from leo.utils.metrics import eval_order_metrics
from leo.utils.metrics import eval_rank_metrics
from leo.utils.order import get_variable_order
from leo.utils.order import get_variable_rank
from .trainer import Trainer

log = logging.getLogger(__name__)


class SVMRankTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None, ps=None, rs=None):
        super().__init__(data, model, cfg, ps, rs)
        # SVMLight dataset files
        self.train_data_file = self.data['train', 'dataset']
        self.val_data_file = self.data['val', 'dataset']
        self.test_data_file = self.data['test', 'dataset']
        # n_items files
        self.train_n_items_file = self.data['train', 'n_items']
        self.val_n_items_file = self.data['val', 'n_items']
        self.test_n_items_file = self.data['test', 'n_items']
        # names files
        self.train_names_file = self.data['train', 'names']
        self.val_names_file = self.data['val', 'names']
        self.test_names_file = self.data['test', 'names']
        # Load result store
        if self.rs is None:
            self.rs = self._get_results_store()
            self.rs['task'] = self.cfg.task
            self.rs['model_name'] = self.cfg.model.name
        # Load prediction store
        if self.ps is None:
            self.ps = self._get_preds_store()
            self.ps['train']['names'] = self.train_names_file.read_text().strip().split('\n')
            self.ps['val']['names'] = self.val_names_file.read_text().strip().split('\n')
            self.ps['test']['names'] = self.test_names_file.read_text().strip().split('\n')

            self.ps['train']['n_items'] = list(map(int, self.train_n_items_file.read_text().strip().split('\n')))
            self.ps['val']['n_items'] = list(map(int, self.train_n_items_file.read_text().strip().split('\n')))
            self.ps['test']['n_items'] = list(map(int, self.test_n_items_file.read_text().strip().split('\n')))

    def run(self):
        # Train
        learn = path.bin / 'svm_rank_learn'
        # model_path = self.res_path / f'pretrained/{self.cfg.problem.name}/{self.cfg.problem.size}'
        model_path = self._get_path('pretrained')
        model_path.mkdir(parents=True, exist_ok=True)
        model = model_path / f'svm_rank_c-{self.cfg.model.c}.dat'

        self.rs['time']['training'] = time.time()
        os.system(f'{learn} -c {self.cfg.model.c} {self.train_data_file} {model}')
        self.rs['time']['training'] = time.time() - self.rs['time']['training']

        log.info(f"  {self.cfg.model.name} train time: {self.rs['time']['training']} \n")

        # Predict on train set
        self.rs['time']['prediction']['train'] = time.time()
        train_pred_file = self._get_split_score(split='train')
        self.rs['time']['prediction']['train'] = time.time() - self.rs['time']['prediction']['train']

        # Predict on val set
        self.rs['time']['prediction']['val'] = time.time()
        val_pred_file = self._get_split_score(split='val')
        self.rs['time']['prediction']['val'] = time.time() - self.rs['time']['prediction']['val']

        train_score, train_n_items = self.unflatten_data_from_file(self.train_data_file, self.train_n_items_file)
        self.ps['train']['score'], _ = self.unflatten_data_from_file(train_pred_file, self.train_n_items_file)

        val_score, val_n_items = self.unflatten_data_from_file(self.val_data_file, self.val_n_items_file)
        self.ps['val']['score'], _ = self.unflatten_data_from_file(val_pred_file, self.val_n_items_file)

        train_order = get_variable_order(scores=train_score, reverse=True)
        self.ps['train']['order'] = get_variable_order(scores=self.ps['train']['score'], reverse=True)
        val_order = get_variable_order(scores=val_score, reverse=True)
        self.ps['val']['order'] = get_variable_order(scores=self.ps['val']['score'], reverse=True)

        train_rank = get_variable_rank(scores=train_score, reverse=True, high_to_low=True)
        self.ps['train']['rank'] = get_variable_rank(scores=self.ps['train']['score'], reverse=True, high_to_low=True)
        val_rank = get_variable_rank(scores=val_score, reverse=True, high_to_low=True)
        self.ps['val']['rank'] = get_variable_rank(scores=self.ps['val']['score'], reverse=True, high_to_low=True)

        # Train set
        log.info('** Train metrics:')
        self.rs['train']['ranking'].extend(eval_order_metrics(train_order,
                                                              self.ps['train']['order'],
                                                              train_n_items))
        self.rs['train']['ranking'].extend(eval_rank_metrics(train_rank,
                                                             self.ps['train']['rank'],
                                                             train_n_items))
        df_train = pd.DataFrame(self.rs['train']['ranking'], columns=['id', 'metric_type', 'metric_value'])
        self.print_rank_metrics(df_train)

        # Validation set
        self.rs['val']['ranking'].extend(eval_order_metrics(val_order,
                                                            self.ps['val']['order'],
                                                            val_n_items))
        self.rs['val']['ranking'].extend(eval_rank_metrics(val_rank,
                                                           self.ps['val']['rank'],
                                                           val_n_items))
        df_val = pd.DataFrame(self.rs['val']['ranking'], columns=['id', 'metric_type', 'metric_value'])
        self.print_rank_metrics(df_train)

        self.val_tau = df_val.query("metric_type == 'kendall-coeff'")['metric_value'].mean()

        self._save_predictions()
        self._save_results()

    def predict(self, split='test'):
        self.rs['time']['prediction'][split] = time.time()
        pred_file = self._get_split_score(split=split)
        self.rs['time']['prediction'][split] = time.time() - self.rs['time']['prediction'][split]

        data_file, n_items_file = getattr(self, '{}_data_file'.format(split)), \
            getattr(self, '{}_n_items_file'.format(split))

        self.ps[split]['score'], _ = self.unflatten_data_from_file(pred_file, n_items_file)
        self.ps[split]['order'] = get_variable_order(scores=self.ps[split]['score'], reverse=True)
        self.ps[split]['rank'] = get_variable_rank(scores=self.ps[split]['score'], reverse=True, high_to_low=True)

        self._save_predictions()
        self._save_results()

    def _get_split_score(self, split='test'):
        classify = path.bin / 'svm_rank_classify'
        model_path = self._get_path('pretrained')
        model = model_path / f'svm_rank_c-{self.cfg.model.c}.dat'

        # data = self.data / f'{self.cfg.problem.size}_dataset_pair_rank_{split}.dat'
        predictions = self._get_path('prediction') / f'svm_rank_c-{self.cfg.model.c}_{split}.dat'

        os.system(f"{classify} {self.data[split, 'dataset']} {model} {predictions}")

        return predictions

    @staticmethod
    def unflatten_data_from_file(filepath, n_item_path):
        n_items = list(map(int, open(n_item_path, 'r').read().strip().split('\n')))
        scores = [float(l.split(' ')[0]) for l in open(filepath, 'r').read().strip().split('\n')]

        _scores = []
        i = 0
        for n_item in n_items:
            _scores.append(scores[i: i + n_item])
            i = i + n_item

        return _scores, n_items
