import json
import logging
import pickle as pkl
import random
import zipfile
from operator import itemgetter
from pathlib import Path
from subprocess import Popen, PIPE, TimeoutExpired

import hydra
import numpy as np

log = logging.getLogger(__name__)


def parse_instance_data(raw_data):
    data = {'value': [], 'weight': [], 'capacity': 0}

    n_vars = int(raw_data.readline())
    n_objs = int(raw_data.readline())
    for _ in range(n_objs):
        data['value'].append(list(map(int, raw_data.readline().split())))
    data['weight'].extend(list(map(int, raw_data.readline().split())))
    data['capacity'] = int(raw_data.readline().split()[0])

    return data


def get_variable_score(data, feature_weights):
    weight, value = np.asarray(data['weight']), np.asarray(data['value'])
    n_items = weight.shape[0]
    value_mean = np.mean(value, axis=0)
    value_max = np.max(value, axis=0)
    value_min = np.min(value, axis=0)

    scores = np.zeros(n_items)
    for fk, fv in feature_weights.items():
        if fk == 'weight':
            scores += fv * weight
        elif fk == 'avg_value':
            scores += fv * value_mean
        elif fk == 'max_value':
            scores += fv * value_max
        elif fk == 'min_value':
            scores += fv * value_min
        elif fk == 'avg_value_by_weight':
            scores += fv * (value_mean / weight)
        elif fk == 'max_value_by_weight':
            scores += fv * (value_max / weight)
        elif fk == 'min_value_by_weight':
            scores += fv * (value_min / weight)

    return scores


def get_variable_order(data, feature_weights):
    """Returns array of variable order
    For example: [2, 1, 0]
    Here 2 is the index of item which should be used first to create the BDD
    """
    n_items = len(data['weight'])
    scores = get_variable_score(data, feature_weights)

    idx_score = [(i, v) for i, v in zip(np.arange(n_items), scores)]
    idx_score.sort(key=itemgetter(1), reverse=True)

    order = [i for i, v in idx_score]

    return order, idx_score


def run_bdd_builder(instance, order, binary, time_limit=60):
    # Prepare the call string to binary
    order_string = " ".join(map(str, order))
    print(order_string)

    cmd = f"{binary}/multiobj {instance} {len(order)} {order_string}"
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
            result = [-1] * 10
    except TimeoutExpired:
        status = "TIMEOUT"
        result = [60] * 10

    return status, result


def perturb_variable_ordering(order,
                              start_idx,
                              end_idx,
                              rng,
                              n_perturbations=1,
                              perturb_type='adjacent',
                              random_seed=100):
    new_order = order.copy()

    log.info(f'{perturb_type}, Start: {start_idx}, End: {end_idx}')
    if perturb_type == 'reverse':
        new_order[:end_idx] = new_order[:end_idx][::-1]

        return new_order

    if perturb_type == 'random':
        random_order = list(np.arange(end_idx))
        random.seed(random_seed)
        random.shuffle(random_order)

        return random_order

    for _ in range(n_perturbations):
        i, j = 0, 1
        if perturb_type == 'adjacent':
            i = rng.randint(start_idx, end_idx)
            i = i - 1 if i == end_idx - 1 else i
            j = i + 1

        elif perturb_type == 'any':
            i = rng.randint(start_idx, end_idx)
            while True:
                j = rng.randint(start_idx, end_idx)
                if i != j:
                    break

        log.info(f'{perturb_type}, Swapped indices: {i}, {j}')

        temp = new_order[i].copy()
        new_order[i] = new_order[j].copy()
        new_order[j] = temp

    return new_order


@hydra.main(version_base='1.1', config_path='../config', config_name='perturbation_analysis.yaml')
def main(cfg):
    # rng = np.random.RandomState(cfg.seed)
    rng = np.random.RandomState()
    result_store = []

    resource_path = Path(__file__).parent.parent / 'resources'
    # inst_path = resource_path.joinpath(f'instances/{cfg.problem.name}')
    smac_output_path = resource_path / f'smac_output/{cfg.problem.name}_so.zip'

    # Zip objects
    # inst_zip_obj = zipfile.ZipFile(inst_path, mode='r')
    smac_output_zip_obj = zipfile.ZipFile(smac_output_path, mode='r')

    # Zip path objects
    # inst_zip_path = zipfile.Path(inst_zip_obj).joinpath(cfg.problem.name)
    inst_zip_path = resource_path.joinpath(f'instances/{cfg.problem.name}')
    smac_output_zip_path = zipfile.Path(smac_output_zip_obj).joinpath(cfg.problem.name)

    from_pid = 0 if cfg.from_pid is None else cfg.from_pid
    to_pid = np.infty if cfg.to_pid is None else cfg.to_pid
    for size in smac_output_zip_path.iterdir():
        if cfg.problem.size is None or (cfg.problem.size is not None and cfg.problem.size == size.name):
            for split in size.iterdir():
                if cfg.problem.split is None or (cfg.problem.split is not None and cfg.problem.split == split.name):
                    for inst in split.iterdir():
                        # Do not process pids, outside from_pid and to_pid
                        pid = int(inst.name.split('_')[-1])
                        if pid < from_pid or pid > to_pid - 1:
                            continue

                        log.info(f'Processing: {inst.name}')

                        # Get instance
                        dat_path = inst_zip_path.joinpath(f'{size.name}/{split.name}/{inst.name}.dat')
                        # raw_data = inst_zip_path.open('r')
                        raw_data = open(str(dat_path), 'r')
                        inst_data = parse_instance_data(raw_data)

                        traj_path = inst.joinpath(f'run_{cfg.seed}/traj.json')
                        if not traj_path.exists():
                            log.info(traj_path, ' does not exist')
                            continue
                        # Get property weight
                        traj = traj_path.open('r')
                        lines = traj.readlines()
                        property_weight = json.loads(lines[-1])
                        order, _ = get_variable_order(inst_data, property_weight['incumbent'])

                        """
                        # Get runtime of best order
                        result_orig = []
                        for _ in range(cfg.n_repeat):
                            status, result = run_bdd_builder(str(dat_path), order, str(resource_path))
                            result_orig.append(result)
                        """

                        # Get result of perturbed order
                        new_order = perturb_variable_ordering(order, cfg.perturb.start_idx, cfg.perturb.end_idx, rng,
                                                              n_perturbations=cfg.perturb.times,
                                                              perturb_type=cfg.perturb.type,
                                                              random_seed=cfg.random_seed)

                        for rid in range(cfg.n_repeat):
                            status, result = run_bdd_builder(str(dat_path), new_order, str(resource_path))
                            temp = [inst.name, rid, cfg.perturb.type, cfg.perturb.times, cfg.perturb.start_idx,
                                    cfg.perturb.end_idx, cfg.random_seed]
                            temp.extend(result)
                            temp.extend([property_weight['cost'], ",".join((map(str, order))),
                                         ",".join((map(str, new_order)))])

                            result_store.append(temp)

    if cfg.save_results:
        pkl.dump(result_store,
                 open(f'pa_{cfg.problem.name}_{cfg.random_seed}'
                      f'_{cfg.perturb.type}_{cfg.perturb.times}_{cfg.perturb.start_idx}'
                      f'_{cfg.perturb.end_idx}_{cfg.n_repeat}_{cfg.from_pid}_{cfg.to_pid}.pkl', 'wb'))


if __name__ == '__main__':
    main()
