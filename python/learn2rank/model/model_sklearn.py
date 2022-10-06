import pickle as pkl

import sklearn.linear_model as sklm


class LinearRegression:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.model = sklm.LinearRegression()

    def train(self, x, y, sample_weight=None):
        self.model.fit(x, y, sample_weight=sample_weight)

    def __call__(self, x):
        return self.model.predict(x)

    def save(self):
        with open('./model.pkl', 'wb') as p:
            pkl.dump(self.model, p)
