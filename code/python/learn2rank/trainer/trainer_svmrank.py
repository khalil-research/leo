import os
from pathlib import Path

from learn2rank.utils.metrics import eval_rank_metrics
from learn2rank.utils.order import score2rank
from .trainer import Trainer


class SVMRankTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None):
        super().__init__(data, model, cfg)
        self.res_path = Path(self.cfg.res_path[self.cfg.machine])
        self.bin_path = self.res_path / 'svm_rank_bin'

        self.data = Path(data)

        self.train_data_file = self.data / f'{self.cfg.problem.name}_dataset_pair_svmrank_train.dat'
        self.val_data_file = self.data / f'{self.cfg.problem.name}_dataset_pair_svmrank_val.dat'

        self.train_n_items_file = self.data / f'{self.cfg.problem.name}_n_items_pair_svmrank_train.dat'
        self.val_n_items_file = self.data / f'{self.cfg.problem.name}_n_items_pair_svmrank_val.dat'

    def run(self):
        self.train()
        train_pred_file = self.predict(split='train')
        val_pred_file = self.predict(split='val')

        train, train_n_items = self.unflatten_data_from_file(self.train_data_file, self.train_n_items_file)
        train_pred, _ = self.unflatten_data_from_file(train_pred_file, self.train_n_items_file)

        val, val_n_items = self.unflatten_data_from_file(self.val_data_file, self.val_n_items_file)
        val_pred, _ = self.unflatten_data_from_file(val_pred_file, self.val_n_items_file)

        train_rank, train_pred_rank = score2rank(train, reverse=True), score2rank(train_pred, reverse=True)
        val_rank, val_pred_rank = score2rank(val, reverse=True), score2rank(val_pred, reverse=True)

        eval_rank_metrics(train_rank, train_pred_rank, train_n_items)
        eval_rank_metrics(val_rank, val_pred_rank, val_n_items)

    def train(self):
        # Train
        learn = self.bin_path / 'svm_rank_learn'
        model = self.res_path / 'pretrained/svm_rank' / f'c-{self.cfg.model.c}.dat'
        os.system(f'{learn} -c {self.cfg.model.c} {self.train_data_file} {model}')

    def predict(self, split='train'):
        classify = self.bin_path / 'svm_rank_classify'
        model = self.res_path / 'pretrained/svm_rank' / f'c-{self.cfg.model.c}.dat'

        data = self.data / f'{self.cfg.problem.name}_dataset_pair_svmrank_{split}.dat'
        predictions = self.res_path / 'predictions/svm_rank' / f'c-{self.cfg.model.c}_{split}.dat'

        os.system(f'{classify} {data} {model} {predictions}')

        return predictions

    @staticmethod
    def unflatten_data_from_file(filepath, n_item_path):
        n_items = list(map(int, open(n_item_path, 'r').read().strip().split('\n')))
        scores = [float(l.split(' ')[0]) for l in open(filepath, 'r').read().strip().split('\n')]
        print(len(scores))

        _scores = []
        i = 0
        for n_item in n_items:
            _scores.append(scores[i: i + n_item])
            i = i + n_item

        return _scores, n_items

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
