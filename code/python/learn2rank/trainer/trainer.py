import pickle
from abc import ABC, abstractmethod


class Trainer(ABC):
    def __init__(self, data, model, config):
        self.data = data
        self.model = model
        self.cfg = config

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def _save_predictions(self):
        with open(f'./prediction_{self.model.id}.pkl', 'wb') as p:
            pickle.dump(self.ps, p)

    def _save_results(self):
        with open(f'./results_{self.model.id}.pkl', 'wb') as p:
            pickle.dump(self.rs, p)
