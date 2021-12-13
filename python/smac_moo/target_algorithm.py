import argparse
import logging
from subprocess import PIPE, Popen, TimeoutExpired

import numpy as np


def run_target_algorithm(opts, log=True):
    if log:
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__file__)
        logger.debug(opts.instance)

    # Prepare the call string to binary
    cmd = f"./multiobj {opts.instance} ./dummy.csv 1"
    cmd += f" {opts.max_weight} {opts.min_weight}" \
           f" {opts.max_avg_value} {opts.min_avg_value}" \
           f" {opts.max_max_value} {opts.min_max_value}" \
           f" {opts.max_min_value} {opts.min_min_value}" \
           f" {opts.max_avg_value_by_weight} {opts.max_max_value_by_weight}"
    if log:
        logging.debug(cmd)

    status = "SUCCESS"
    runtime = 0
    try:
        io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE)
        # Call target algorithm with cutoff time
        (stdout_, stderr_) = io.communicate(timeout=opts.cutoff)

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
        runtime = opts.cutoff

    # Output result for SMAC.
    print(f"Result for SMAC: {status}, {runtime}, 0, 0, {opts.seed}")


# def tae_runner(cfg, seed, instance):
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Run config
    parser.add_argument('instance', type=str, help='Path to the instance file')
    parser.add_argument('specifics', type=str, help='Specifics')
    parser.add_argument('cutoff', type=int, help='Cutoff runtime for the target algorithm')
    parser.add_argument('runlength', type=int, help='Runlength')
    parser.add_argument('seed', help='Random seed')
    # Solver Config. Always start with a `-` to their name
    parser.add_argument('-max_weight', type=float, default=0)
    parser.add_argument('-min_weight', type=float, default=1)
    parser.add_argument('-max_avg_value', type=float, default=0)
    parser.add_argument('-min_avg_value', type=float, default=0)
    parser.add_argument('-max_max_value', type=float, default=0)
    parser.add_argument('-min_max_value', type=float, default=0)
    parser.add_argument('-max_min_value', type=float, default=0)
    parser.add_argument('-min_min_value', type=float, default=0)
    parser.add_argument('-max_avg_value_by_weight', type=float, default=0)
    parser.add_argument('-max_max_value_by_weight', type=float, default=0)
    opts = parser.parse_args()

    run_target_algorithm(opts)
