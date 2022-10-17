import argparse
import logging
import sys
from operator import itemgetter
from pathlib import Path
from subprocess import Popen, PIPE, TimeoutExpired

import numpy as np


# Maximal virtual memory for subprocesses (in bytes).
# MAX_VIRTUAL_MEMORY = 1 * 1024 * 1024 * 1024  # 1 GB
#
#
# def limit_virtual_memory():
#     # Maximal virtual memory for subprocesses (in bytes).
#     # MAX_VIRTUAL_MEMORY = 1 * 1024 * 1024 * 1024  # 1 GB
#     global MAX_VIRTUAL_MEMORY
#
#     # The tuple below is of the form (soft limit, hard limit). Limit only
#     # the soft part so that the limit can be increased later (setting also
#     # the hard limit would prevent that).
#     # When the limit cannot be changed, setrlimit() raises ValueError.
#     resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY, MAX_VIRTUAL_MEMORY))
#

def run_bdd_builder(instance, order, time_limit=60, mem_limit=16, do_log=None, logger=None):
    # Prepare the call string to binary
    order_string = " ".join(map(str, order))
    bin_path = Path(__file__).parent.parent / 'resources/multiobj'
    cmd = f"{bin_path} {instance} {len(order)} {order_string}"
    if do_log:
        logger.debug(cmd)

    # Maximal virtual memory for subprocesses (in bytes).
    # global MAX_VIRTUAL_MEMORY
    # MAX_VIRTUAL_MEMORY = mem_limit * (1024 ** 3)

    status = "SUCCESS"
    runtime = 0
    try:
        io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE)
        # Call target algorithm with cutoff time
        (stdout_, stderr_) = io.communicate(timeout=time_limit)
        if do_log:
            logger.debug(stdout_)

        # Decode and parse output
        stdout, stderr = stdout_.decode('utf-8'), stderr_.decode('utf-8')
        if len(stdout) and "Solved" in stdout:
            # Sum the last three floating points to calculate the total time
            # This is binary dependent and can change
            runtime = np.sum(list(map(float, stdout.strip().split(',')[-3:])))
        else:
            # If the instance is not solved successfully on the cluster, we either hit the
            # runtime limit or memory limit. In either of the two cases, we will not be
            # allowed to run more instances. Hence, we stop the parameter optimization
            # process using the ABORT signal
            status = "ABORT"
            runtime = time_limit
    except TimeoutExpired:
        status = "TIMEOUT"
        runtime = time_limit

    return status, runtime


def read_from_file(filepath):
    data = {'value': [], 'weight': [], 'capacity': 0}

    with open(filepath, 'r') as fp:
        n_vars = int(fp.readline())
        n_objs = int(fp.readline())

        for _ in range(n_objs):
            data['value'].append(list(map(int, fp.readline().split())))

        data['weight'].extend(list(map(int, fp.readline().split())))

        data['capacity'] = int(fp.readline().split()[0])

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


def run_target_algorithm(instance, cutoff, feature_weights, log=True):
    logger = None
    if log:
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__file__)
        logger.debug(instance)

    # num_objectives = int(instance.split("/")[-1].split("_")[2])
    data = read_from_file(instance)
    order, _ = get_variable_order(data, feature_weights)

    status, runtime = run_bdd_builder(instance, order, time_limit=cutoff, do_log=log, logger=logger)
    print(f"Result for SMAC: {status}, {runtime}, 0, 0, 0")


# def tae_runner(cfg, seed, instance):
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Read in first 5 arguments.
    instance = sys.argv[1]
    specifics = sys.argv[2]
    cutoff = int(float(sys.argv[3]) + 1)
    runlength = int(sys.argv[4])
    seed = int(sys.argv[5])

    # Read in parameter setting and build a dictionary mapping param_name to param_value.
    params = sys.argv[6:]
    feature_weights = dict((name[1:], float(value)) for name, value in zip(params[::2], params[1::2]))

    run_target_algorithm(instance, cutoff, feature_weights)
