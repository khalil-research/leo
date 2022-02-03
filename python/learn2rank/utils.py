import resource
from operator import itemgetter
from subprocess import Popen, PIPE, TimeoutExpired

import numpy as np

# Maximal virtual memory for subprocesses (in bytes).
MAX_VIRTUAL_MEMORY = 1 * 1024 * 1024 * 1024  # 1 GB


def get_rank(sorted_data):
    idx_rank = {}
    for rank, item in enumerate(sorted_data):
        idx_rank[item[0]] = rank

    return idx_rank


def read_from_file(p, filepath):
    data = {'value': [], 'weight': [], 'capacity': 0}

    with open(filepath, 'r') as fp:
        fp.readline()
        fp.readline()

        for _ in range(p):
            data['value'].append(list(map(int, fp.readline().split())))

        data['weight'].extend(list(map(int, fp.readline().split())))

        data['capacity'] = int(fp.readline().split()[0])

    return data


def get_static_orders(data):
    order = {
        'max_weight': None,
        'min_weight': None,
        'max_avg_profit': None,
        'min_avg_profit': None,
        'max_max_profit': None,
        'min_max_profit': None,
        'max_min_profit': None,
        'min_min_profit': None,
        'max_avg_profit_by_weight': None,
        'max_max_profit_by_weight': None,
    }

    n_items = len(data['weight'])
    for o in order.keys():
        if o == 'max_weight':
            idx_weight = [(i, w) for i, w in enumerate(data['weight'])]
            # print(o, idx_weight)
            idx_weight.sort(key=itemgetter(1), reverse=True)
            order[o] = [i[0] for i in idx_weight]

        elif o == 'min_weight':
            idx_weight = [(i, w) for i, w in enumerate(data['weight'])]
            idx_weight.sort(key=itemgetter(1))
            order[o] = [i[0] for i in idx_weight]

        elif o == 'max_avg_profit':
            mean_profit = np.mean(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(mean_profit)]
            idx_profit.sort(key=itemgetter(1), reverse=True)
            order[o] = [i[0] for i in idx_profit]

        elif o == 'min_avg_profit':
            mean_profit = np.mean(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(mean_profit)]
            idx_profit.sort(key=itemgetter(1))
            order[o] = [i[0] for i in idx_profit]

        elif o == 'max_max_profit':
            max_profit = np.max(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(max_profit)]
            idx_profit.sort(key=itemgetter(1), reverse=True)
            order[o] = [i[0] for i in idx_profit]

        elif o == 'min_max_profit':
            max_profit = np.max(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(max_profit)]
            idx_profit.sort(key=itemgetter(1))
            order[o] = [i[0] for i in idx_profit]

        elif o == 'max_min_profit':
            min_profit = np.min(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(min_profit)]
            idx_profit.sort(key=itemgetter(1), reverse=True)
            order[o] = [i[0] for i in idx_profit]

        elif o == 'min_min_profit':
            min_profit = np.min(data['value'], 0)
            idx_profit = [(i, mp) for i, mp in enumerate(min_profit)]
            idx_profit.sort(key=itemgetter(1))
            order[o] = [i[0] for i in idx_profit]

        elif o == 'max_avg_profit_by_weight':
            mean_profit = np.mean(data['value'], 0)
            profit_by_weight = [v / w for v, w in zip(mean_profit, data['weight'])]
            idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
            idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)
            # print(idx_profit_by_weight)
            order[o] = [i[0] for i in idx_profit_by_weight]

        elif o == 'max_max_profit_by_weight':
            max_profit = np.max(data['value'], 0)
            profit_by_weight = [v / w for v, w in zip(max_profit, data['weight'])]
            idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
            idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)
            # print(idx_profit_by_weight)
            order[o] = [i[0] for i in idx_profit_by_weight]

    return order


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
