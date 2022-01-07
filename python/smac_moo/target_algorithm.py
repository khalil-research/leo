import argparse
import logging
import sys
from subprocess import PIPE, Popen, TimeoutExpired

import numpy as np


def run_target_algorithm(instance, cutoff, opts, log=True):
    if log:
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__file__)
        logger.debug(instance)

    # Prepare the call string to binary
    cmd = f"./multiobj {instance}"
    cmd += f' {opts["weight"]}' \
           f' {opts["avg_value"]}' \
           f' {opts["max_value"]}' \
           f' {opts["min_value"]}' \
           f' {opts["avg_value_by_weight"]}' \
           f' {opts["max_value_by_weight"]}' \
           f' {opts["min_value_by_weight"]}'
    if log:
        logging.debug(cmd)

    status = "SUCCESS"
    runtime = 0
    try:
        io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE)
        # Call target algorithm with cutoff time
        (stdout_, stderr_) = io.communicate(timeout=cutoff)
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
    except TimeoutExpired:
        status = "TIMEOUT"
        runtime = cutoff

    # Output result for SMAC.
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
    opts = dict((name[1:], value) for name, value in zip(params[::2], params[1::2]))

    run_target_algorithm(instance, cutoff, opts)
