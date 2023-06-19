from learn2rank.utils import Factory
from .ann import NeuralRankingMachine
from .model_baseline import Canonical
from .model_baseline import MinWeight
from .model_baseline import SmacAll
from .model_baseline import SmacOne
from .model_rank import GradientBoostingRanker
from .model_rank import SVMRank
from .model_sklearn import DecisionTreeRegressor
from .model_sklearn import GradientBoostingRegressor
from .model_sklearn import Lasso
from .model_sklearn import LinearRegression
from .model_sklearn import Ridge

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
model_factory.register_member('Canonical', Canonical)
