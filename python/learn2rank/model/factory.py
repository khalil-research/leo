from sklearn.linear_model import LinearRegression

from learn2rank.utils import Factory
from .ann import NeuralRankingMachine

model_factory = Factory()
model_factory.register_member('NeuralRankingMachine', NeuralRankingMachine)
model_factory.register_member('LR', LinearRegression)
