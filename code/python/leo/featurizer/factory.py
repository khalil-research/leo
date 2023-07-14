from leo.utils import Factory
from .knapsack import KnapsackFeaturizer

featurizer_factory = Factory()
featurizer_factory.register_member('knapsack', KnapsackFeaturizer)
