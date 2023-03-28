import os
from pathlib import Path

from .trainer import Trainer


class SVMRankTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None):
        super().__init__(data, model, cfg)
        self.res_path = Path(self.cfg.res_path[self.cfg.machine])
        self.bin_path = self.res_path / 'svm_rank_bin'
        self.data = Path(data)

    def run(self):
        learn = self.bin_path / 'svm_rank_learn'
        data = self.data / 'train.dat'
        model = self.bin_path / 'pretrained/svm_rank' / f'c-{self.cfg.model.c}.dat'

        os.system(f'{learn} -c {self.cfg.model.c} {data} {model}')

    def predict(self, split='train'):
        classify = self.bin_path / 'svm_rank_classify'
        data = self.data / f'{split}.dat'
        model = self.bin_path / 'pretrained/svm_rank' / f'c-{self.cfg.model.c}.dat'
        predictions = self.data / 'predictions/svm_rank' / f'c-{self.cfg.model.c}.dat'

        os.system(f'{classify} {data} {model} {predictions}')
