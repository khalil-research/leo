import argparse
import os
from pathlib import Path

from utils import get_config_space, smac_factory


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
    output_dir = str(opts.output_dir.joinpath(Path(instances[0][0]).stem))
    scenario_dict['output_dir'] = output_dir

    smac = smac_factory(scenario_dict, output_dir, opts)
    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    print("Optimized configuration %s" % str(incumbent))


def main(opts):
    logger = get_logger(opts.verbose) if opts.verbose else None

    # Total number of slurm workers detected
    # Defaults to 1 if not running under SLURM
    N_WORKERS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

    # This worker's array index. Assumes slurm array job is zero-indexed
    # Defaults to zero if not running under SLURM
    this_worker = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))

    # Create configuration space
    cs = get_config_space()

    # Define scenario
    base_scenario_dict = {
        "algo": "python target_algorithm.py",
        "cs": cs,
        "deterministic": "true",
        "run_obj": "runtime",
        "cutoff_time": opts.cutoff_time,
        "wallclock_limit": opts.wallclock_limit,
    }

    # Check paths
    opts.data_path = Path(opts.data_path)
    opts.output_dir = Path(opts.output_dir)
    assert opts.data_path.is_dir()

    # Prepare training dataset over which SMAC will optimize
    all_files = [p for p in opts.data_path.iterdir()]
    all_files_prefix = "_".join(all_files[0].stem.split("_")[:-1])
    files = [str(opts.data_path.joinpath(all_files_prefix + f"_{i}.dat"))
             for i in range(opts.num_instances)]

    if opts.mode == 'one':
        dataset = [[files[i]] for i in range(this_worker, opts.num_instances, N_WORKERS)]
        for instance in dataset:
            # [[instance]]
            run_smac([instance], base_scenario_dict, opts)
    else:
        dataset = [[file] for file in files]
        # [[instance_1], [instance_2]]
        run_smac(dataset, base_scenario_dict, opts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='one', choices=['one', 'all'])
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--cutoff_time', type=int, default=1)
    parser.add_argument('--wallclock_limit', type=int, default=30)
    parser.add_argument('--data_path', type=str, default='../../data/kp/3_40/train/')
    parser.add_argument('--num_instances', type=int, default=1)
    parser.add_argument('--output_dir', type=str,
                        default='./output/smac3-output_3_40_test')
    parser.add_argument('--restore_run', type=int, default=0)
    # Should be more than the wallclock limit of the last run
    parser.add_argument('--rwcl', type=int, default=20, help='Wallclock limit for restored run')
    parser.add_argument('--rods', type=str, default='+', help='Suffix of output dir for restored run')
    parser.add_argument('--verbose', type=int, default=1)

    opts = parser.parse_args()
    main(opts)
