import os
from pathlib import Path

from learn2rank.utils.metrics import eval_order_metrics
from learn2rank.utils.metrics import eval_rank_metrics
from learn2rank.utils.order import pred_score2order
from learn2rank.utils.order import pred_score2rank
from .trainer import Trainer


class SVMRankTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None, ps=None, rs=None):
        super().__init__(data, model, cfg, ps, rs)
        self.bin_path = self.res_path / 'svm_rank_bin'

        self.data = Path(data)

        self.train_data_file = self.data / f'{self.cfg.problem.size}_dataset_pair_svmrank_train.dat'
        self.val_data_file = self.data / f'{self.cfg.problem.size}_dataset_pair_svmrank_val.dat'

        self.train_n_items_file = self.data / f'{self.cfg.problem.size}_n_items_pair_svmrank_train.dat'
        self.val_n_items_file = self.data / f'{self.cfg.problem.size}_n_items_pair_svmrank_val.dat'

        self.train_names_file = self.data / f'{self.cfg.problem.size}_names_pair_svmrank_train.dat'
        self.val_names_file = self.data / f'{self.cfg.problem.size}_names_pair_svmrank_val.dat'

        if self.rs is None:
            self.rs = self._get_results_store()
            self.rs['task'] = self.cfg.task
            self.rs['model_name'] = self.cfg.model.name

        if self.ps is None:
            self.ps = self._get_preds_store()
            self.ps['train']['names'] = self.train_names_file.read_text().strip().split('\n')
            self.ps['val']['names'] = self.val_names_file.read_text().strip().split('\n')
            self.ps['train']['n_items'] = list(map(int, self.train_n_items_file.read_text().strip().split('\n')))
            self.ps['val']['n_items'] = list(map(int, self.train_n_items_file.read_text().strip().split('\n')))

    def run(self):
        self.train()
        train_pred_file = self.predict(split='train')
        val_pred_file = self.predict(split='val')

        train_score, train_n_items = self.unflatten_data_from_file(self.train_data_file, self.train_n_items_file)
        self.ps['train']['score'], _ = self.unflatten_data_from_file(train_pred_file, self.train_n_items_file)

        val_score, val_n_items = self.unflatten_data_from_file(self.val_data_file, self.val_n_items_file)
        self.ps['val']['score'], _ = self.unflatten_data_from_file(val_pred_file, self.val_n_items_file)

        # High score -> higher place in the order
        train_order = pred_score2order(train_score, reverse=True)
        self.ps['train']['order'] = pred_score2order(self.ps['train']['score'], reverse=True)
        val_order = pred_score2order(val_score, reverse=True)
        self.ps['val']['order'] = pred_score2order(self.ps['val']['score'], reverse=True)

        # High score -> high rank
        train_rank = pred_score2rank(train_score, reverse=True)
        self.ps['train']['rank'] = pred_score2rank(self.ps['train']['score'], reverse=True)
        val_rank = pred_score2rank(val_score, reverse=True)
        self.ps['val']['rank'] = pred_score2rank(self.ps['val']['score'], reverse=True)

        # Train set
        self.rs['train']['ranking'].extend(eval_order_metrics(train_order,
                                                              self.ps['train']['order'],
                                                              train_n_items))
        self.rs['train']['ranking'].extend(eval_rank_metrics(train_rank,
                                                             self.ps['train']['rank'],
                                                             train_n_items))

        # Validation set
        self.rs['val']['ranking'].extend(eval_order_metrics(val_order,
                                                            self.ps['val']['order'],
                                                            val_n_items))
        self.rs['val']['ranking'].extend(eval_rank_metrics(val_rank,
                                                           self.ps['val']['rank'],
                                                           val_n_items))

        self._save_predictions()
        self._save_results()

    def train(self):
        # Train
        learn = self.bin_path / 'svm_rank_learn'

        model_path = self.res_path / f'pretrained/{self.cfg.problem.name}/{self.cfg.problem.size}'
        model_path.mkdir(parents=True, exist_ok=True)
        model = model_path / f'svm_rank_c-{self.cfg.model.c}.dat'

        os.system(f'{learn} -c {self.cfg.model.c} {self.train_data_file} {model}')

    def predict(self, split='test'):
        classify = self.bin_path / 'svm_rank_classify'

        model_path = self.res_path / f'pretrained/{self.cfg.problem.name}/{self.cfg.problem.size}'
        model = model_path / f'svm_rank_c-{self.cfg.model.c}.dat'

        data = self.data / f'{self.cfg.problem.size}_dataset_pair_svmrank_{split}.dat'
        predictions = self.pred_path / f'svm_rank_c-{self.cfg.model.c}_{split}.dat'
        os.system(f'{classify} {data} {model} {predictions}')

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
