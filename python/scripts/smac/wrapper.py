import sys
import os
import time
import re
from subprocess import Popen, PIPE

# Read in first 5 arguments.
instance = sys.argv[1]
specifics = sys.argv[2]
print(sys.argv[3])
cutoff = int(float(sys.argv[3]) + 1)
runlength = int(sys.argv[4])
seed = int(sys.argv[5])

print(instance, specifics, cutoff, runlength, seed)

# Read in parameter setting and build a dictionary mapping param_name to param_value.
params = sys.argv[6:]
configMap = dict((name, value)
                 for name, value in zip(params[::2], params[1::2]))
print(configMap)
# Construct the call string to Spear.
# spear_binary = "examples/spear/Spear-32_1.2.1"

# Execute the call and track its runtime.
# print(cmd)

# multiobj_binary = "./multiobj"
cmd = f"./multiobj {instance} ./sample_weighted.csv 1"
for name, value in configMap.items():
    cmd += f" {value}"

start_time = time.time()
io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE)
(stdout_, stderr_) = io.communicate()
runtime = time.time() - start_time

# runtime = float(configMap['-min_weight'])/10
# time.sleep(runtime)

# Very simply parsing of Spear's output. Note that in practice we would check the found solution to guard against bugs.
status = "SUCCESS"

# Output result for SMAC.
print(f"Result for SMAC: {status}, {runtime}, 0, 0, {seed}")
