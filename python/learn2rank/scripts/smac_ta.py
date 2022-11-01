import argparse
import logging
import sys
from operator import itemgetter

import numpy as np

from learn2rank.utils.bdd import run_bdd_builder


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

    status, runtime = run_bdd_builder(instance, order, time_limit=cutoff, get_runtime=True)
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
