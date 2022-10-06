from learn2rank.utils import Factory
from .ann import NeuralRankingMachine
from .model_sklearn import LinearRegression

model_factory = Factory()
model_factory.register_member('NeuralRankingMachine', NeuralRankingMachine)
model_factory.register_member('LinearRegression', LinearRegression)
