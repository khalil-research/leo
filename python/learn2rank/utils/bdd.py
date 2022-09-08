import resource
from subprocess import Popen, PIPE, TimeoutExpired

import numpy as np

# Maximal virtual memory for subprocesses (in bytes).
MAX_VIRTUAL_MEMORY = 1 * 1024 * 1024 * 1024  # 1 GB


def limit_virtual_memory():
    # Maximal virtual memory for subprocesses (in bytes).
    # MAX_VIRTUAL_MEMORY = 1 * 1024 * 1024 * 1024  # 1 GB
    global MAX_VIRTUAL_MEMORY

    # The tuple below is of the form (soft limit, hard limit). Limit only
    # the soft part so that the limit can be increased later (setting also
    # the hard limit would prevent that).
    # When the limit cannot be changed, setrlimit() raises ValueError.
    resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY, MAX_VIRTUAL_MEMORY))


def run_bdd_builder(instance, order, time_limit=60, mem_limit=2):
    # Prepare the call string to binary
    order_string = " ".join(map(str, order))
    cmd = f"./multiobj {instance} {len(order)} {order_string}"
    # Maximal virtual memory for subprocesses (in bytes).
    global MAX_VIRTUAL_MEMORY
    MAX_VIRTUAL_MEMORY = mem_limit * (1024 ** 3)

    status = "SUCCESS"
    runtime = 0
    try:
        io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE, preexec_fn=limit_virtual_memory)
        # Call target algorithm with cutoff time
        (stdout_, stderr_) = io.communicate(timeout=time_limit)

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
