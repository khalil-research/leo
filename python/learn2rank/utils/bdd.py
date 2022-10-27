import logging
import resource
from subprocess import Popen, PIPE, TimeoutExpired

import numpy as np

# Maximal virtual memory for subprocesses (in bytes).
MAX_VIRTUAL_MEMORY = 1 * 1024 * 1024 * 1024  # 1 GB
log = logging.getLogger(__name__)


def limit_virtual_memory():
    # Maximal virtual memory for subprocesses (in bytes).
    # MAX_VIRTUAL_MEMORY = 1 * 1024 * 1024 * 1024  # 1 GB
    global MAX_VIRTUAL_MEMORY

    # The tuple below is of the form (soft limit, hard limit). Limit only
    # the soft part so that the limit can be increased later (setting also
    # the hard limit would prevent that).
    # When the limit cannot be changed, setrlimit() raises ValueError.
    resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY, MAX_VIRTUAL_MEMORY))


def run_bdd_builder(instance, order, binary, time_limit=60, mem_limit=None):
    # Prepare the call string to binary
    order_string = " ".join(map(str, order))
    cmd = f"{binary}/multiobj {instance} {len(order)} {order_string}"
    # Maximal virtual memory for subprocesses (in bytes).
    if mem_limit is not None:
        global MAX_VIRTUAL_MEMORY
        MAX_VIRTUAL_MEMORY = mem_limit * (1024 ** 3)

    status = "SUCCESS"
    runtime = 0
    try:
        if mem_limit is None:
            # Do not set memory limit
            io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE)
        else:
            io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE, preexec_fn=limit_virtual_memory)

        # Call target algorithm with cutoff time
        (stdout_, stderr_) = io.communicate(timeout=time_limit)

        # Decode and parse output
        stdout, stderr = stdout_.decode('utf-8'), stderr_.decode('utf-8')
        if len(stdout) and "Solved" in stdout:
            # Sum the last three floating points to calculate the total time
            # This is binary dependent and can change
            result = stdout.split(':')[1]
            result = list(map(float, result.split(',')))

            runtime = np.sum(list(map(float, stdout.strip().split(',')[-3:])))
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
        result = [time_limit] * 10

    return status, result
