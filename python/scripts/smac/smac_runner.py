import time
import numpy as np

from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.configspace import ConfigurationSpace
from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario

instances = [f"./3_20/train/kp_7_3_20_{i}.dat" for i in range(250)]


def main():
    # Define configuration space
    cs = ConfigurationSpace()

    max_weight = UniformFloatHyperparameter(
        "max_weight", 1e-7, 1, default_value=1e-7)

    min_weight = UniformFloatHyperparameter(
        "min_weight", 1e-7, 1, default_value=1)

    max_avg_value = UniformFloatHyperparameter(
        "max_avg_value", 1e-7, 1, default_value=1e-7)

    min_avg_value = UniformFloatHyperparameter(
        "min_avg_value", 1e-7, 1, default_value=1e-7)

    max_max_value = UniformFloatHyperparameter(
        "max_max_value", 1e-7, 1, default_value=1e-7)

    min_max_value = UniformFloatHyperparameter(
        "min_max_value", 1e-7, 1, default_value=1e-7)

    max_min_value = UniformFloatHyperparameter(
        "max_min_value", 1e-7, 1, default_value=1e-7)

    min_min_value = UniformFloatHyperparameter(
        "min_min_value", 1e-7, 1, default_value=1e-7)

    max_avg_value_by_weight = UniformFloatHyperparameter(
        "max_avg_value_by_weight", 1e-7, 1, default_value=1e-7)

    max_max_value_by_weight = UniformFloatHyperparameter(
        "max_max_value_by_weight", 1e-7, 1, default_value=1e-7)

    cs.add_hyperparameters([max_weight,
                            min_weight,
                            max_avg_value,
                            min_avg_value,
                            max_max_value,
                            min_max_value,
                            max_min_value,
                            min_min_value,
                            max_avg_value_by_weight,
                            max_max_value_by_weight])

    # Define scenario
    scenario = Scenario({
        "algo": "python wrapper.py",
        "cs": cs,
        "execdir": ".",
        "deterministic": "true",
        "run_obj": "runtime",
        "cutoff_time": 2,
        "wallclock-limit": 3600,
        "instances": instances
    })

    # Define smac facade
    smac = SMAC4AC(
        scenario=scenario
    )

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    print("Optimized configuration %s" % str(incumbent))


if __name__ == "__main__":
    main()
