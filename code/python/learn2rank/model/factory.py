from learn2rank.utils import Factory
from .ann import NeuralRankingMachine
from .model_baseline import Lex
from .model_baseline import SmacD
from .model_baseline import SmacI
from .model_baseline import HeuristicOrder
from .model_baseline import Random
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
model_factory.register_member('HeuristicWeight', HeuristicOrder)
model_factory.register_member('HeuristicValue', HeuristicOrder)
model_factory.register_member('HeuristicValueByWeight', HeuristicOrder)
model_factory.register_member('SmacI', SmacI)
model_factory.register_member('SmacD', SmacD)
model_factory.register_member('Lex', Lex)
model_factory.register_member('Random', Random)
