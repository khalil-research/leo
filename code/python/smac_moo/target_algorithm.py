import argparse
import logging
import sys

from utils import read_from_file, get_variable_order
from utils import run_bdd_builder


def run_target_algorithm(instance, cutoff, feature_weights, log=True):
    logger = None
    if log:
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__file__)
        logger.debug(instance)

    num_objectives = int(instance.split("/")[-1].split("_")[2])
    data = read_from_file(num_objectives, instance)
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