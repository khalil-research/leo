import logging
import sys
from subprocess import PIPE, Popen, TimeoutExpired

import numpy as np


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__file__)

# Read in first 5 arguments.
instance = sys.argv[1]
specifics = sys.argv[2]
cutoff = int(float(sys.argv[3]) + 1)
runlength = int(sys.argv[4])
seed = int(sys.argv[5])

instance = f"./3_20/train/kp_7_3_20_{instance}.dat"
# logger.debug(instance)

# Read in parameter setting and build a dictionary mapping param_name to param_value.
params = sys.argv[6:]
configMap = dict((name, value)
                 for name, value in zip(params[::2], params[1::2]))
# logger.info(configMap)
# Construct the call string to multiobj
cmd = f"./multiobj {instance} ./dummy.csv 1"
cmd += f" {configMap['-max_weight']} {configMap['-min_weight']}" \
    f" {configMap['-max_avg_value']} {configMap['-min_avg_value']}" \
    f" {configMap['-max_max_value']} {configMap['-min_max_value']}" \
    f" {configMap['-max_min_value']} {configMap['-min_min_value']}" \
    f" {configMap['-max_avg_value_by_weight']} {configMap['-max_max_value_by_weight']}"
# logging.debug(cmd)
status = "SUCCESS"
runtime = 0
try:
    io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE)
    (stdout_, stderr_) = io.communicate(timeout=cutoff)
    stdout, stderr = stdout_.decode('utf-8'), stderr_.decode('utf-8')

    if len(stdout) and "Solved" in stdout:
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
print(f"Result for SMAC: {status}, {runtime}, 0, 0, {seed}")
