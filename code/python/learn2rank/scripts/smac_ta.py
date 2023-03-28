import argparse
import logging
import os
import pathlib
import sys

sys.path.insert(0, os.environ.get('module_path'))

from learn2rank.utils.bdd import run_bdd_builder
from learn2rank.utils.data import read_data_from_file
from learn2rank.utils.order import get_variable_order_from_weights


def run_target_algorithm(instance, cutoff, property_weights, log=True):
    logger = None
    if log:
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__file__)
        logger.debug(instance)

    inst_path = pathlib.Path(instance)
    acronym = inst_path.stem.split('_')[0]
    data = read_data_from_file(acronym, instance)
    order, _ = get_variable_order_from_weights(data, property_weights)

    # Prepare the call string to bin_path
    prob_id = os.environ.get('prob_id')
    preprocess = os.environ.get('preprocess')
    bin_path = os.environ.get('bin_path')
    bin_name = os.environ.get('bin_name')
    mem_limit = float(os.environ.get('mem_limit'))
    mask_mem_limit = bool(int(os.environ.get('mask_mem_limit')))

    status, runtime = run_bdd_builder(instance, order,
                                      prob_id=prob_id, preprocess=preprocess,
                                      bin_path=bin_path, bin_name=bin_name,
                                      time_limit=cutoff, get_runtime=True,
                                      mem_limit=mem_limit, mask_mem_limit=mask_mem_limit)

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
    property_weights = dict((name[1:], float(value)) for name, value in zip(params[::2], params[1::2]))

    run_target_algorithm(instance, cutoff, property_weights)
