from learn2rank.utils import Factory
from .ann import NeuralRankingMachine
from .model_sklearn import DecisionTreeRegressor
from .model_sklearn import GradientBoostingRegressor
from .model_sklearn import Lasso
from .model_sklearn import LinearRegression
from .model_sklearn import Ridge
from .model_xgb import GradientBoostingRanker
from .model_smac import SmacOne
from .model_smac import SmacAll


class SVMRank:
    """Dummy class for SVMRank model"""

    def __init__(self, cfg=None):
        self.cfg = cfg

    @property
    def id(self):
        return f'svmrank_c-{self.cfg.c}'


class MinWeight:
    """Dummy class for Min-weight model"""

    def __init__(self, cfg=None):
        self.cfg = cfg

    @property
    def id(self):
        return 'minwt'


model_factory = Factory()
model_factory.register_member('NeuralRankingMachine', NeuralRankingMachine)
model_factory.register_member('LinearRegression', LinearRegression)
model_factory.register_member('Ridge', Ridge)
model_factory.register_member('Lasso', Lasso)
model_factory.register_member('DecisionTreeRegressor', DecisionTreeRegressor)
model_factory.register_member('GradientBoostingRegressor', GradientBoostingRegressor)
model_factory.register_member('GradientBoostingRanker', GradientBoostingRanker)
model_factory.register_member('SVMRank', SVMRank)
model_factory.register_member('MinWeight', MinWeight)
model_factory.register_member('SmacOne', SmacOne)
model_factory.register_member('SmacAll', SmacAll)

