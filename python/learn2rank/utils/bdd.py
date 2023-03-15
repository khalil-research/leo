import logging
import os
import resource
from subprocess import Popen, PIPE, TimeoutExpired

import numpy as np

log = logging.getLogger(__name__)


def limit_virtual_memory():
    # Maximal virtual memory for subprocesses (in bytes).
    # MAX_VIRTUAL_MEMORY = 1 * 1024 * 1024 * 1024  # 1 GB
    mvm = int(os.environ.get('MAX_VIRTUAL_MEMORY'))
    # The tuple below is of the form (soft limit, hard limit). Limit only
    # the soft part so that the limit can be increased later (setting also
    # the hard limit would prevent that).
    # When the limit cannot be changed, setrlimit() raises ValueError.
    resource.setrlimit(resource.RLIMIT_AS, (mvm, mvm))


def run_bdd_builder(instance, order, prob_id=None, preprocess=None, bin_path=None,
                    bin_name='multiobj', time_limit=60, get_runtime=False, mem_limit=16,
                    mask_mem_limit=False):
    # Prepare the call string to bin_path
    bin_path = os.environ.get('bin_path') if bin_path is None else bin_path
    binary = f'{bin_path}/{bin_name}'
    prob_id = os.environ.get('prob_id') if prob_id is None else prob_id
    preprocess = os.environ.get('preprocess') if preprocess is None else preprocess

    order_string = " ".join(map(str, order))
    cmd = f"{binary} {instance} {prob_id} {preprocess} {len(order)} {order_string}"
    # Maximal virtual memory for subprocesses (in bytes).
    os.environ['MAX_VIRTUAL_MEMORY'] = str(mem_limit) if mem_limit is None else str(mem_limit * (1024 ** 3))
    status = "SUCCESS"

    log.info(f'Executing: {cmd}')
    log.info(f"Memory limit: {os.environ.get('MAX_VIRTUAL_MEMORY')}")
    try:
        preexec_fn = None if mem_limit is None else limit_virtual_memory
        io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE, preexec_fn=preexec_fn)

        # Call target algorithm with cutoff time
        (stdout_, stderr_) = io.communicate(timeout=time_limit)

        # Decode and parse output
        stdout, stderr = stdout_.decode('utf-8'), stderr_.decode('utf-8')
        if len(stdout) and "Solved" in stdout:
            # Sum the last three floating points to calculate the total time
            # This is binary dependent and can change
            blobs = stdout.split('Solved:')[1].split('#')
            # blob = stdout[7:].split("#")
            result = list(map(float, blobs[0].split(',')))
            if get_runtime:
                result = np.sum(result[-3:])
            else:
                result.append(blobs[1].strip())

        else:
            # If the instance is not solved successfully on the cluster, we either hit the
            # runtime limit or memory limit. In either of the two cases, we will not be
            # allowed to run more instances. Hence, we stop the parameter optimization
            # process using the ABORT signal
            if mask_mem_limit:
                # Mask the memory out signal as timeout. To be used for set covering instances
                log.info("MEMOUT/MASKED")

                status = "TIMEOUT"
                result = time_limit if get_runtime else [time_limit] * 11
            else:
                log.info("MEMOUT/ABORT")

                status = "ABORT"
                result = -1 if get_runtime else [-1] * 11

    except TimeoutExpired:
        log.info("TIMEOUT")

        status = "TIMEOUT"
        result = time_limit if get_runtime else [time_limit] * 11

    return status, result
