import logging
import os
import pickle as pkl
import random
from pathlib import Path
from subprocess import Popen, PIPE, TimeoutExpired

import hydra
from omegaconf import DictConfig

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


def make_result_column(split, pid, result, pred='True', run_id=0):
    col = [split, pid, pred, run_id]
    col.extend(result)

    return col


@hydra.main(version_base='1.1', config_path='../config', config_name='eval_ordering.yaml')
def main(cfg: DictConfig):
    resource_path = Path(__file__).parent.parent / 'resources'
    inst_path = resource_path.joinpath(f'instances/{cfg.problem.name}')
    preds_path = resource_path.joinpath(f'predictions/{cfg.pred.name}')
    preds = pkl.load(open(str(preds_path), 'rb'))
    if cfg.split == 'train':
        preds = preds['tr']
    else:
        preds = preds['val']
    # preds = preds['val'] if cfg.split is None else preds[cfg.split]
    results = []

    names, n_items, order = preds['names'], preds['n_items'], preds['order']
    for _name, _n_item, _order in zip(names, n_items, order):
        _, _, n_objs, n_vars, pid = _name.split('_')
        pid = int(pid)
        if pid < cfg.from_pid or pid >= cfg.to_pid:
            continue

        log.info(f'Processing {_name}')

        dat_path = inst_path / f'{n_objs}_{n_vars}/{cfg.split}/{_name}.dat'

        if cfg.pred.switch:
            status, result = run_bdd_builder(str(dat_path), _order[:_n_item], str(resource_path),
                                             time_limit=cfg.bdd.timelimit)
            results.append(make_result_column(cfg.split, pid, result))

        if cfg.random.switch:
            for run_id, s in enumerate(cfg.random.seed):
                random_order = list(range(_n_item))
                random.seed(s)
                random.shuffle(random_order)
                status, result = run_bdd_builder(str(dat_path), random_order, str(resource_path),
                                                 time_limit=cfg.bdd.timelimit)
                results.append(make_result_column(cfg.split, pid, result, pred='False', run_id=run_id))

    with open(f"eval_{cfg.pred.name.split('.')[0]}_{cfg.split}_{cfg.from_pid}_{cfg.to_pid}.pkl", 'wb') as fp:
        pkl.dump(results, fp)


if __name__ == '__main__':
    main()
