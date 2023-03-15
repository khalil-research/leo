import os
from pathlib import Path

import hydra
import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.callbacks import IncorporateRunResultCallback
from smac.configspace import ConfigurationSpace
from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario


class TestCallback(IncorporateRunResultCallback):
    def __init__(self):
        self.num_call = 0
        self.smbo = None
        self.run_info = None
        self.result = None
        self.time_left = None

    def __call__(self, smbo, run_info, result, time_left):
        self.num_call += 1
        self.smbo = smbo
        self.run_info = run_info
        self.result = result
        self.time_left = time_left

        if self.num_call == 2:
            return False


def get_config_space(init_incumbent, problem=None, size=None, width=None):
    cs = ConfigurationSpace()

    # Hyperparams
    weight = UniformFloatHyperparameter("weight", -width.default, width.default)
    avg_value = UniformFloatHyperparameter("avg_value", -width.default, width.default)
    max_value = UniformFloatHyperparameter("max_value", -width.default, width.default)
    min_value = UniformFloatHyperparameter("min_value", -width.default, width.default)
    avg_value_by_weight = UniformFloatHyperparameter("avg_value_by_weight", -width.default, width.default)
    max_value_by_weight = UniformFloatHyperparameter("max_value_by_weight", -width.default, width.default)
    min_value_by_weight = UniformFloatHyperparameter("min_value_by_weight", -width.default, width.default)

    if problem == 'setcovering' and width.label < 1:
        label = UniformFloatHyperparameter("label", 1 - width.label, 1)
    elif problem == 'setcovering' and width.label == 1:
        label = UniformFloatHyperparameter("label", -1, 1)

    # Fetch incumbent config and initialize hyperparams
    incb = init_incumbent.split('/')
    if incb[0] == 'smac_optimized':
        from learn2rank.prop_wt import optimized as prop_wt_opt
        assert problem in prop_wt_opt and size in prop_wt_opt[problem]
        pwts = prop_wt_opt[problem][size][incb[1]]
    else:
        from learn2rank.prop_wt import static as prop_wt_static
        pwts = prop_wt_static[incb[0]]

    weight.default_value = pwts['weight']
    avg_value.default_value = pwts['avg_value']
    max_value.default_value = pwts['max_value']
    min_value.default_value = pwts['min_value']
    avg_value_by_weight.default_value = pwts['avg_value_by_weight']
    max_value_by_weight.default_value = pwts['max_value_by_weight']
    min_value_by_weight.default_value = pwts['min_value_by_weight']
    props_lst = [weight, avg_value, max_value, min_value, avg_value_by_weight,
                 max_value_by_weight, min_value_by_weight]

    if problem == 'setcovering':
        label.default_value = pwts['label']
        props_lst.append(label)

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
    scenario_dict = base_scenario_dict.copy()
    # Optimize over one instance
    scenario_dict['instances'] = instances
    # logger.debug('*******************')
    # logger.debug(load)

    # Set output directory
    output_dir = str(Path(instances[0][0]).stem)
    scenario_dict['output_dir'] = output_dir

    # dummy_callback = TestCallback()
    scenario = Scenario(scenario_dict)
    smac = SMAC4AC(
        scenario=scenario,
        rng=np.random.RandomState(opts.seed),
        run_id=opts.seed
    )
    # smac.register_callback(dummy_callback)

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    print("Optimized configuration %s" % str(incumbent))

    # return dummy_callback


@hydra.main(version_base='1.1', config_path='../config', config_name='smac_runner.yaml')
def main(cfg):
    logger = get_logger(cfg.verbose) if cfg.verbose else None

    # Set environment variables
    os.environ['module_path'] = cfg.module_path[cfg.machine]
    os.environ['bin_path'] = cfg.res_path[cfg.machine]
    os.environ['bin_name'] = cfg.bin_name
    os.environ['prob_id'] = str(cfg.problem.id)
    os.environ['preprocess'] = str(cfg.problem.preprocess)

    # Create configuration space
    cs = get_config_space(cfg.init_incumbent, problem=cfg.problem.name, size=cfg.problem.size, width=cfg.width)

    # Define scenario
    base_scenario_dict = {
        "cs": cs,
        "deterministic": "true",
        "run_obj": "runtime",
        "cutoff_time": cfg.cutoff_time,
        "wallclock_limit": cfg.wallclock_limit
    }
    # "rand_prob": 0.5,
    # "maxR": 1,
    # "intens_adaptive_capping_slackfactork": 2

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

    cb = None
    if cfg.mode == 'one':
        dataset = [[files[i]] for i in range(cfg.num_instances)]
        for instance in dataset:
            # [[instance]]
            cb = run_smac([instance], base_scenario_dict, cfg)
    else:
        dataset = [[file] for file in files]
        # [[instance_1], [instance_2]]
        cb = run_smac(dataset, base_scenario_dict, cfg)

    if cb is not None:
        print(cb.run_info)
        print(cb.run_info.capped)
        print(cb.result)
        print(cb.time_left)


if __name__ == "__main__":
    main()
