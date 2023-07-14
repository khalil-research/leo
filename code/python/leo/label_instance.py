import os
from pathlib import Path

import hydra
import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.configspace import ConfigurationSpace
from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario

from leo import path


def get_config_space(width=None):
    cs = ConfigurationSpace()

    # Hyperparams initialized with min_weight
    weight = UniformFloatHyperparameter("weight", -width, width, default_value=-1)
    avg_value = UniformFloatHyperparameter("avg_value", -width, width, default_value=0)
    max_value = UniformFloatHyperparameter("max_value", -width, width, default_value=0)
    min_value = UniformFloatHyperparameter("min_value", -width, width, default_value=0)
    avg_value_by_weight = UniformFloatHyperparameter("avg_value_by_weight", -width, width, default_value=0)
    max_value_by_weight = UniformFloatHyperparameter("max_value_by_weight", -width, width, default_value=0)
    min_value_by_weight = UniformFloatHyperparameter("min_value_by_weight", -width, width, default_value=0)
    props_lst = [weight, avg_value, max_value, min_value, avg_value_by_weight,
                 max_value_by_weight, min_value_by_weight]
    # Add hyperparams to config store
    cs.add_hyperparameters(props_lst)

    return cs


def get_logger(verbosity):
    import logging
    logging.basicConfig(level=logging.INFO if verbosity == 1 else logging.DEBUG)
    logger = logging.getLogger(__file__)

    return logger


def run_smac(instances, base_scenario_dict, opts):
    assert len(instances)

    # Create scenario
    scenario_dict = base_scenario_dict.copy()
    scenario_dict['instances'] = instances
    scenario_dict['output_dir'] = path[opts.mode] / opts.problem.name / opts.problem.size
    scenario_dict['output_dir'] /= opts.split / str(Path(instances[0][0]).stem)
    scenario = Scenario(scenario_dict)

    # Create SMAC object
    smac = SMAC4AC(
        scenario=scenario,
        rng=np.random.RandomState(opts.seed),
        run_id=opts.seed)

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    print('Optimized configuration %s' % str(incumbent))


@hydra.main(version_base='1.2', config_path='./config', config_name='label_instance.yaml')
def main(cfg):
    logger = get_logger(cfg.verbose) if cfg.verbose else None

    # Set environment variables
    os.environ['module_path'] = str(path.module)
    os.environ['bin_path'] = str(path.bin)
    os.environ['bin_name'] = cfg.bin_name
    os.environ['prob_id'] = str(cfg.problem.id)
    os.environ['preprocess'] = str(cfg.problem.preprocess)
    os.environ['mem_limit'] = str(cfg.mem_limit)

    # Create configuration space
    cs = get_config_space(width=cfg.width)
    # Define scenario
    base_scenario_dict = {
        'cs': cs,
        'deterministic': 'true',
        'run_obj': 'runtime',
        'algo': f'python {Path(__file__).parent}/smac_worker.py',
        'cutoff_time': cfg.cutoff_time,
        'wallclock_limit': cfg.wallclock_limit
    }

    print(path.module)
    # Check paths
    data_path = path.instances / cfg.problem.inst_name / cfg.problem.size / cfg.split
    assert data_path.is_dir()
    prefix = f'kp_7_{cfg.problem.size}'
    files = [str(data_path.joinpath(prefix + f"_{i}.dat"))
             for i in range(cfg.from_pid, cfg.from_pid + cfg.num_instances)]

    if cfg.mode == 'SmacI':
        # Hack to provide one instance dataset to smac
        dataset = [[files[i]] for i in range(cfg.num_instances)]
        for instance in dataset:
            # [[instance]]
            run_smac([instance], base_scenario_dict, cfg)
    elif cfg.mode == 'SmacD':
        dataset = [[file] for file in files]
        # [[instance_1], [instance_2]]
        run_smac(dataset, base_scenario_dict, cfg)
    else:
        raise ValueError(f'Invalid smac mode: {cfg.mode}')


if __name__ == '__main__':
    main()
