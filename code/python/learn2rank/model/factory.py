from learn2rank.utils import Factory
from .ann import NeuralRankingMachine
from .model_sklearn import DecisionTreeRegressor
from .model_sklearn import GradientBoostingRegressor
from .model_sklearn import Lasso
from .model_sklearn import LinearRegression
from .model_sklearn import Ridge


class SVMRank:
    """Dummy class for SVMRank model"""

    def __init__(self, cfg=None):
        self.cfg = cfg


model_factory = Factory()
model_factory.register_member('NeuralRankingMachine', NeuralRankingMachine)
model_factory.register_member('LinearRegression', LinearRegression)
model_factory.register_member('Ridge', Ridge)
model_factory.register_member('Lasso', Lasso)
model_factory.register_member('DecisionTreeRegressor', DecisionTreeRegressor)
model_factory.register_member('GradientBoostingRegressor', GradientBoostingRegressor)
model_factory.register_member('SVMRank', SVMRank)
