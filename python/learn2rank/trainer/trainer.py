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
    def predict(self, x):
        pass
