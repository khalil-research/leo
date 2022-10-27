import logging
import os
import pickle as pkl
from pathlib import Path
from subprocess import Popen, PIPE, TimeoutExpired

import hydra
import numpy as np
from omegaconf import DictConfig

from learn2rank.utils.order import get_static_order

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


def run_bdd_builder(instance, order, binary, time_limit=60):
    # Prepare the call string to binary
    order_string = " ".join(map(str, order))
    print(order_string)

    cmd = f"{binary}/multiobj {instance} {len(order)} {order_string}"
    print(cmd)
    status = "SUCCESS"
    try:
        io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE)
        # Call target algorithm with cutoff time
        (stdout_, stderr_) = io.communicate(timeout=time_limit)

        # Decode and parse output
        stdout, stderr = stdout_.decode('utf-8'), stderr_.decode('utf-8')
        if len(stdout) and "Solved" in stdout:
            # Sum the last three floating points to calculate the total time
            # This is binary dependent and can change
            result = stdout.split(':')[1]
            result = list(map(float, result.split(',')))
        else:
            # If the instance is not solved successfully on the cluster, we either hit the
            # runtime limit or memory limit. In either of the two cases, we will not be
            # allowed to run more instances. Hence, we stop the parameter optimization
            # process using the ABORT signal
            status = "ABORT"
            log.info("ABORT")

            result = [-1] * 10
    except TimeoutExpired:
        log.info("TIMEOUT")
        status = "TIMEOUT"
        result = [60] * 10

    return status, result


def make_result_column(split, pid, result, order_type, run_id=0):
    col = [split, pid, order_type, run_id]
    col.extend(result)

    return col


@hydra.main(version_base='1.1', config_path='../config', config_name='eval_static_ordering.yaml')
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
        order = get_static_order(data, cfg.order_type)

        status, result = run_bdd_builder(str(dat_path), order, str(resource_path),
                                         time_limit=cfg.bdd.timelimit)
        log.info(f'Status: {status}')
        if status == 'SUCCESS':
            log.info(f'Solving time: {np.sum(result[-3:])}')

        results.append(make_result_column(cfg.split, pid, result, cfg.order_type))

    with open(f"eval_static_{cfg.problem.size}_{cfg.split}_{cfg.from_pid}_{cfg.to_pid}.pkl", 'wb') as fp:
        pkl.dump(results, fp)


if __name__ == '__main__':
    main()
