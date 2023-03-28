import os
from pathlib import Path

import hydra

from learn2rank.smac_runner import smac_runner_factory


def get_logger(verbosity):
    import logging
    logging.basicConfig(level=logging.INFO if verbosity == 1 else logging.DEBUG)
    logger = logging.getLogger(__file__)

    return logger


def create_datasets(mode, files, num_instances):
    if mode == 'one':
        datasets = [[[files[i]]] for i in range(num_instances)]
        assert len(datasets) == num_instances
    else:
        datasets = [[[file] for file in files]]
        assert len(datasets) == 1

    return datasets


@hydra.main(version_base='1.1', config_path='../config', config_name='smac_runner.yaml')
def main(cfg):
    logger = get_logger(cfg.verbose) if cfg.verbose else None

    # Set environment variables
    os.environ['module_path'] = cfg.module_path[cfg.machine]
    os.environ['bin_path'] = cfg.res_path[cfg.machine]
    os.environ['bin_name'] = cfg.bin_name
    os.environ['prob_id'] = str(cfg.problem.id)
    os.environ['preprocess'] = str(cfg.problem.preprocess)
    os.environ['mem_limit'] = str(cfg.mem_limit)
    os.environ['mask_mem_limit'] = str(1 if cfg.mask_mem_limit else 0)

    # Check paths
    resource_path = Path(cfg.res_path[cfg.machine])
    data_path = resource_path / f'instances/{cfg.problem.inst_name}/{cfg.problem.size}/{cfg.split}'
    assert data_path.is_dir()

    # Prepare training dataset over which SMAC will optimize
    all_files = [p for p in data_path.iterdir()]
    # For arranging them from 0 to num_instances
    all_files_prefix = "_".join(all_files[0].stem.split("_")[:-1])
    files = [str(data_path.joinpath(all_files_prefix + f"_{i}.dat"))
             for i in range(cfg.from_pid, cfg.from_pid + cfg.num_instances)]
    datasets = create_datasets(cfg.mode, files, cfg.num_instances)

    for dataset in datasets:
        smac_runner = smac_runner_factory.create(cfg.problem.name, cfg=cfg)
        smac_runner.run(dataset)


if __name__ == '__main__':
    main()
