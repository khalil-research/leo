import argparse
from pathlib import Path

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.configspace import ConfigurationSpace
from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario

# logger = logging.getLogger('runner')
# instances = [f"./3_20/train/kp_7_3_20_{i}.dat" for i in range(250)]
instances = [f"{i}" for i in range(250)]


def load_incumbent(old_output_dir, new_scenario):
    from smac.utils.io.traj_logging import TrajLogger

    traj_path = old_output_dir / "traj_aclib2.json"
    trajectory = TrajLogger.read_traj_aclib_format(
        fn=traj_path,
        cs=new_scenario.cs)
    incumbent = trajectory[-1]["incumbent"]

    return incumbent


def load_stats(old_output_dir, new_scenario):
    from smac.stats.stats import Stats

    stats_path = old_output_dir / "stats.json"
    stats = Stats(new_scenario)
    stats.load(str(stats_path))

    return stats


def load_runhistory(old_output_dir, new_scenario):
    from smac.runhistory.runhistory import RunHistory

    rh_path = old_output_dir / "runhistory.json"
    runhistory = RunHistory()
    runhistory.load_json(str(rh_path), new_scenario.cs)

    return runhistory


def get_config_space():
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

    return cs


def main(args):
    # Set logging 0: Off, 1: INFO, 2: DEBUG
    if args.verbose:
        import logging
        if args.verbose == 1:
            logging.basicConfig(level=logging.INFO)
        elif args.verbose == 2:
            logging.basicConfig(level=logging.INFO)

    # Create configuration space
    cs = get_config_space()

    # Define scenario
    scenario_dict = {
        "algo": "python wrapper.py",
        "cs": cs,
        "execdir": ".",
        "deterministic": "true",
        "run_obj": "runtime",
        "cutoff_time": args.cutoff_time,
        "wallclock_limit": args.wallclock_limit,
        "instances": instances,
        "output_dir": args.output_dir
    }
    scenario = Scenario(scenario_dict)

    if args.restore_run:
        p = Path(args.output_dir) / f"run_{args.seed}"
        if p.exists():
            cmd_options = {'wallclock_limit': scenario_dict['wallclock_limit']*2,  # overwrite these args
                           'output_dir': scenario_dict['output_dir']+"+"}
            new_scenario = Scenario(scenario_dict, cmd_options=cmd_options)

            # We load the runhistory
            runhistory = load_runhistory(p, new_scenario)
            # And the stats
            stats = load_stats(p, new_scenario)
            # And the trajectory
            incumbent = load_incumbent(p, new_scenario)

            # Define smac facade
            smac = SMAC4AC(
                scenario=scenario,
                runhistory=runhistory,
                stats=stats,
                restore_incumbent=incumbent,
                rng=np.random.RandomState(args.seed),
                run_id=args.seed
            )
    else:
        smac = SMAC4AC(
            scenario=scenario,
            rng=np.random.RandomState(args.seed),
            run_id=args.seed
        )

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    print("Optimized configuration %s" % str(incumbent))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--cutoff_time', type=int, default=3)
    parser.add_argument('--wallclock_limit', type=int, default=120)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--output_dir', type=str,
                        default='./smac3-output-3_20')
    parser.add_argument('--restore_run', type=int, default=0)
    args = parser.parse_args()
    main(args)
