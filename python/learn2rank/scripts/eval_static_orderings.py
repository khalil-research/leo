import logging
import os
import pickle as pkl
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from learn2rank.utils.bdd import run_bdd_builder
from learn2rank.utils.order import get_static_order
from learn2rank.utils.order import make_result_column

# A logger for this file
log = logging.getLogger(__name__)

print(os.getcwd())


def parse_instance_data(raw_data):
    data = {'value': [], 'weight': [], 'capacity': 0}

    n_vars = int(raw_data.readline())
    n_objs = int(raw_data.readline())
    for _ in range(n_objs):
        data['value'].append(list(map(int, raw_data.readline().split())))
    data['weight'].extend(list(map(int, raw_data.readline().split())))
    data['capacity'] = int(raw_data.readline().split()[0])

    return data


@hydra.main(version_base='1.1', config_path='../config', config_name='eval_static_orderings.yaml')
def main(cfg: DictConfig):
    resource_path = Path(__file__).parent.parent / 'resources'
    inst_path = resource_path.joinpath(f'instances/{cfg.problem.name}')
    # preds = preds['val'] if cfg.split is None else preds[cfg.split]
    results = []

    inst_path = inst_path / cfg.problem.size / cfg.split
    for dat_path in inst_path.rglob('*.dat'):
        pid = int(dat_path.stem.split('_')[-1])
        if pid < cfg.from_pid or pid >= cfg.to_pid:
            continue

        log.info(f'Processing: {dat_path.name}')
        log.info(f'Order type: {cfg.order_type}')
        raw_data = open(dat_path, 'r')
        data = parse_instance_data(raw_data)
        orders = get_static_order(data, cfg.order_type)

        for run_id, order in enumerate(orders):
            print(type(cfg.bdd.memlimit), cfg.bdd.memlimit)
            status, result = run_bdd_builder(str(dat_path), order, binary=str(resource_path),
                                             time_limit=cfg.bdd.timelimit, mem_limit=cfg.bdd.memlimit)
            log.info(f'Status: {status}')
            if status == 'SUCCESS':
                log.info(f'Solving time: {np.sum(result[-3:]):.3f}')

            results.append(make_result_column(cfg.problem.name,
                                              cfg.problem.size,
                                              cfg.split,
                                              pid,
                                              cfg.order_type,
                                              result,
                                              run_id=run_id))

    with open(f"eval_static_{cfg.problem.size}_{cfg.split}_{cfg.from_pid}_{cfg.to_pid}.pkl", 'wb') as fp:
        pkl.dump(results, fp)


if __name__ == '__main__':
    main()
