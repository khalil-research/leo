import os
from pathlib import Path

import hydra
import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.configspace import ConfigurationSpace
from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario


def get_knapsack_config_space():
    # Define knapsack configuration space
    cs = ConfigurationSpace()

    weight = UniformFloatHyperparameter(
        "weight", -1, 1, default_value=-1)

    avg_value = UniformFloatHyperparameter(
        "avg_value", -1, 1, default_value=0)

    max_value = UniformFloatHyperparameter(
        "max_value", -1, 1, default_value=0)

    min_value = UniformFloatHyperparameter(
        "min_value", -1, 1, default_value=0)

    avg_value_by_weight = UniformFloatHyperparameter(
        "avg_value_by_weight", -1, 1, default_value=0)

    max_value_by_weight = UniformFloatHyperparameter(
        "max_value_by_weight", -1, 1, default_value=0)

    min_value_by_weight = UniformFloatHyperparameter(
        "min_value_by_weight", -1, 1, default_value=0)

    cs.add_hyperparameters([weight,
                            avg_value,
                            max_value,
                            min_value,
                            avg_value_by_weight,
                            max_value_by_weight,
                            min_value_by_weight])

    return cs


def get_setpacking_config_space():
    # Define knapsack configuration space
    cs = ConfigurationSpace()

    weight = UniformFloatHyperparameter(
        "weight", -1, 1, default_value=0)

    avg_value = UniformFloatHyperparameter(
        "avg_value", -1, 1, default_value=0)

    max_value = UniformFloatHyperparameter(
        "max_value", -1, 1, default_value=0)

    min_value = UniformFloatHyperparameter(
        "min_value", -1, 1, default_value=0)

    avg_value_by_weight = UniformFloatHyperparameter(
        "avg_value_by_weight", -1, 1, default_value=0)

    max_value_by_weight = UniformFloatHyperparameter(
        "max_value_by_weight", -1, 1, default_value=0)

    min_value_by_weight = UniformFloatHyperparameter(
        "min_value_by_weight", -1, 1, default_value=0)

    cs.add_hyperparameters([weight,
                            avg_value,
                            max_value,
                            min_value,
                            avg_value_by_weight,
                            max_value_by_weight,
                            min_value_by_weight])

    return cs


def get_setcovering_config_space():
    # Define binproblem configuration space
    cs = ConfigurationSpace()

    weight = UniformFloatHyperparameter(
        "weight", -1, 1, default_value=0)

    # weight_Av_mean = UniformFloatHyperparameter(
    #     "weight", -1, 1, default_value=-1)

    avg_value = UniformFloatHyperparameter(
        "avg_value", -1, 1, default_value=0)

    max_value = UniformFloatHyperparameter(
        "max_value", -1, 1, default_value=0)

    min_value = UniformFloatHyperparameter(
        "min_value", -1, 1, default_value=0)

    avg_value_by_weight = UniformFloatHyperparameter(
        "avg_value_by_weight", -1, 1, default_value=0)

    max_value_by_weight = UniformFloatHyperparameter(
        "max_value_by_weight", -1, 1, default_value=0)

    min_value_by_weight = UniformFloatHyperparameter(
        "min_value_by_weight", -1, 1, default_value=0)

    # dot = UniformFloatHyperparameter(
    #     "min_value_by_weight", -1, 1, default_value=0)

    cs.add_hyperparameters([weight,
                            avg_value,
                            max_value,
                            min_value,
                            avg_value_by_weight,
                            max_value_by_weight,
                            min_value_by_weight])

    return cs


config_space = {
    'knapsack': get_knapsack_config_space(),
    'setcovering': get_setcovering_config_space(),
    'setpacking': get_setpacking_config_space()
}


def get_logger(verbosity):
    import logging
    logging.basicConfig(level=logging.INFO if verbosity == 1 else logging.DEBUG)
    logger = logging.getLogger(__file__)

    return logger


def run_smac(instances, base_scenario_dict, opts):
    assert len(instances)
    scenario_dict = base_scenario_dict.copy()
    # Optimize over one instance
    scenario_dict['instances'] = instances
    # logger.debug('*******************')
    # logger.debug(load)

    # Set output directory
    output_dir = str(Path(instances[0][0]).stem)
    scenario_dict['output_dir'] = output_dir

    scenario = Scenario(scenario_dict)
    smac = SMAC4AC(
        scenario=scenario,
        rng=np.random.RandomState(opts.seed),
        run_id=opts.seed
    )
    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    print("Optimized configuration %s" % str(incumbent))


@hydra.main(version_base='1.1', config_path='../config', config_name='smac_runner.yaml')
def main(cfg):
    logger = get_logger(cfg.verbose) if cfg.verbose else None

    # Set environment variables
    os.environ['module_path'] = cfg.module_path[cfg.machine]
    os.environ['bin_path'] = cfg.res_path[cfg.machine]
    os.environ['prob_id'] = str(cfg.problem.id)
    os.environ['preprocess'] = str(cfg.problem.preprocess)

    # Create configuration space
    cs = config_space.get(cfg.problem.name)
    assert cs is not None
    # cs = get_config_space(cfg.problem.name)

    # Define scenario
    base_scenario_dict = {
        "cs": cs,
        "deterministic": "true",
        "run_obj": "runtime",
        "cutoff_time": cfg.cutoff_time,
        "wallclock_limit": cfg.wallclock_limit,
    }

    # Check paths
    resource_path = Path(cfg.res_path[cfg.machine])
    data_path = resource_path / f'instances/{cfg.problem.inst_name}/{cfg.problem.size}/{cfg.split}'
    assert data_path.is_dir()

    # python_path = f'{Path(__file__).parent}/smac_ta.py
    # base_scenario_dict['algo'] = f"python {str(python_path)}/learn2rank/scripts/smac_ta.py"
    base_scenario_dict['algo'] = f"python {Path(__file__).parent}/smac_ta.py"
    # Prepare training dataset over which SMAC will optimize
    all_files = [p for p in data_path.iterdir()]
    # For arranging them from 0 to num_instances
    all_files_prefix = "_".join(all_files[0].stem.split("_")[:-1])
    files = [str(data_path.joinpath(all_files_prefix + f"_{i}.dat"))
             for i in range(cfg.from_pid, cfg.from_pid + cfg.num_instances)]

    if cfg.mode == 'one':
        dataset = [[files[i]] for i in range(cfg.num_instances)]
        for instance in dataset:
            # [[instance]]
            run_smac([instance], base_scenario_dict, cfg)
    else:
        dataset = [[file] for file in files]
        # [[instance_1], [instance_2]]
        run_smac(dataset, base_scenario_dict, cfg)


if __name__ == "__main__":
    main()
