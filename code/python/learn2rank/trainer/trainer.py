import pickle
from abc import ABC, abstractmethod
from pathlib import Path


class Trainer(ABC):
    def __init__(self, data, model, config):
        self.data = data
        self.model = model
        self.cfg = config

        self.rs = {}
        self.ps = {}

        self.res_path = Path(self.cfg.res_path[self.cfg.machine])
        if self.cfg.dataset.fused:
            self.pred_path = self.res_path / f'predictions/{self.cfg.problem.name}'
        else:
            self.pred_path = self.res_path / f'predictions/{self.cfg.problem.name}/{self.cfg.problem.size}'
            
        self.pred_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def _save_predictions(self):
        out_path = self.pred_path / f'prediction_{self.model.id}.pkl'
        with open(out_path, 'wb') as p:
            pickle.dump(self.ps, p)

    def _save_results(self):
        out_path = self.pred_path / f'results_{self.model.id}.pkl'
        with open(out_path, 'wb') as p:
            pickle.dump(self.rs, p)
