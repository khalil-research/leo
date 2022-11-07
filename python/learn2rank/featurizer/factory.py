from learn2rank.utils import Factory
from .binproblem import BinproblemFeaturizer
from .knapsack import KnapsackFeaturizer

featurizer_factory = Factory()
featurizer_factory.register_member('knapsack', KnapsackFeaturizer)
featurizer_factory.register_member('binproblem', BinproblemFeaturizer)
