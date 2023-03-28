from learn2rank.utils import Factory
from .kp_smac_runner import KnapsackSMACRunner
from .sc_smac_runner import SetcoverSMACRunner

smac_runner_factory = Factory()
smac_runner_factory.register_member('knapsack', KnapsackSMACRunner)
smac_runner_factory.register_member('setcovering', SetcoverSMACRunner)
