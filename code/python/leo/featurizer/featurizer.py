class Featurizer:
    def __init__(self, cfg):
        self.cfg = cfg

    def get(self, *args, **kwargs):
        raise NotImplementedError
