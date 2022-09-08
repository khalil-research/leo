import resource
from operator import itemgetter
from subprocess import Popen, PIPE, TimeoutExpired

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.configspace import ConfigurationSpace
from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario

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


def load_incumbent(old_output_dir, new_scenario):
    from smac.utils.io.traj_logging import TrajLogger

    traj_path = old_output_dir / "traj_aclib2.json"
    trajectory = TrajLogger.read_traj_aclib_format(
        fn=traj_path,
        cs=new_scenario.cs)
    incumbent = trajectory[-1]["incumbent"]

    return incumbent


def load_stats(old_output_dir, new_scenario):
    from smac.stats.stats import Stats

    stats_path = old_output_dir / "stats.json"
    stats = Stats(new_scenario)
    stats.load(str(stats_path))

    return stats


def load_runhistory(old_output_dir, new_scenario):
    from smac.runhistory.runhistory import RunHistory

    rh_path = old_output_dir / "runhistory.json"
    runhistory = RunHistory()
    runhistory.load_json(str(rh_path), new_scenario.cs)

    return runhistory


def get_config_space():
    # Define configuration space
    cs = ConfigurationSpace()

    weight = UniformFloatHyperparameter(
        "weight", -1, 1, default_value=-1)

    avg_value = UniformFloatHyperparameter(
        "avg_value", -1, 1, default_value=0)

    max_value = UniformFloatHyperparameter(
        "max_value", -1, 1, default_value=0)

    min_value = UniformFloatHyperparameter(
        "min_value", -1, 1, default_value=0)

    avg_value_by_weight = UniformFloatHyperparameter(
        "avg_value_by_weight", -1, 1, default_value=0)

    max_value_by_weight = UniformFloatHyperparameter(
        "max_value_by_weight", -1, 1, default_value=0)

    min_value_by_weight = UniformFloatHyperparameter(
        "min_value_by_weight", -1, 1, default_value=0)

    cs.add_hyperparameters([weight,
                            avg_value,
                            max_value,
                            min_value,
                            avg_value_by_weight,
                            max_value_by_weight,
                            min_value_by_weight])

    return cs


def read_from_file(num_objectives, filepath):
    data = {'value': [], 'weight': [], 'capacity': 0}

    with open(filepath, 'r') as fp:
        fp.readline()
        fp.readline()

        for _ in range(num_objectives):
            data['value'].append(list(map(int, fp.readline().split())))

        data['weight'].extend(list(map(int, fp.readline().split())))

        data['capacity'] = int(fp.readline().split()[0])

    return data


def get_orderings(data):
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
            print(o, idx_weight)
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
            order[o] = [i[0] for i in idx_profit_by_weight]

        elif o == 'max_max_profit_by_weight':
            max_profit = np.max(data['value'], 0)
            profit_by_weight = [v / w for v, w in zip(max_profit, data['weight'])]
            idx_profit_by_weight = [(i, f) for i, f in enumerate(profit_by_weight)]
            idx_profit_by_weight.sort(key=itemgetter(1), reverse=True)
            order[o] = [i[0] for i in idx_profit_by_weight]

    return order


def get_variable_score(data, feature_weights):
    weight, value = np.asarray(data['weight']), np.asarray(data['value'])
    n_items = weight.shape[0]
    value_mean = np.mean(value, axis=0)
    value_max = np.max(value, axis=0)
    value_min = np.min(value, axis=0)

    scores = np.zeros(n_items)
    for fk, fv in feature_weights.items():
        if fk == 'weight':
            scores += fv * weight
        elif fk == 'avg_value':
            scores += fv * value_mean
        elif fk == 'max_value':
            scores += fv * value_max
        elif fk == 'min_value':
            scores += fv * value_min
        elif fk == 'avg_value_by_weight':
            scores += fv * (value_mean / weight)
        elif fk == 'max_value_by_weight':
            scores += fv * (value_max / weight)
        elif fk == 'min_value_by_weight':
            scores += fv * (value_min / weight)

    return scores


def get_variable_order(data, feature_weights):
    """Returns array of variable order
    For example: [2, 1, 0]
    Here 2 is the index of item which should be used first to create the BDD
    """
    n_items = len(data['weight'])
    scores = get_variable_score(data, feature_weights)

    idx_score = [(i, v) for i, v in zip(np.arange(n_items), scores)]
    idx_score.sort(key=itemgetter(1), reverse=True)

    order = [i for i, v in idx_score]

    return order, idx_score


def get_variable_rank(data, feature_weights):
    """Returns array of variable ranks
    For example: [2, 1, 0]
    Item 0 must be used third to construct the BDD
    """
    n_items = len(data['weight'])

    _, idx_score = get_variable_order(data, feature_weights)

    variable_rank = np.zeros(n_items)
    for rank, (i, _) in enumerate(idx_score):
        variable_rank[i] = rank

    return variable_rank


def smac_factory(scenario_dict, output_dir, opts):
    # p = Path(output_dir) / f"run_{opts.seed}"
    # if opts.restore_run and p.exists():
    #     cmd_options = {'wallclock_limit': opts.rwcl,  # overwrite these args
    #                    'output_dir': scenario_dict['output_dir'] + opts.rods}
    #     new_scenario = Scenario(scenario_dict, cmd_options=cmd_options)
    #
    #     # We load the runhistory
    #     runhistory = load_runhistory(p, new_scenario)
    #     # And the stats
    #     stats = load_stats(p, new_scenario)
    #     # And the trajectory
    #     incumbent = load_incumbent(p, new_scenario)
    #
    #     # Define smac facade
    #     # logger.debug('* Restoring smac')
    #     smac = SMAC4AC(
    #         scenario=new_scenario,
    #         runhistory=runhistory,
    #         stats=stats,
    #         restore_incumbent=incumbent,
    #         rng=np.random.RandomState(opts.seed),
    #         run_id=opts.seed,
    #         n_jobs=opts.n_jobs
    #     )
    # else:
    scenario = Scenario(scenario_dict)
    # logger.debug('* New smac')
    smac = SMAC4AC(
        scenario=scenario,
        rng=np.random.RandomState(opts.seed),
        run_id=opts.seed
    )

    return smac


def run_bdd_builder(instance, order, time_limit=60, mem_limit=16, do_log=None, logger=None):
    # Prepare the call string to binary
    order_string = " ".join(map(str, order))
    cmd = f"./multiobj {instance} {len(order)} {order_string}"
    if do_log:
        logger.debug(cmd)

    # Maximal virtual memory for subprocesses (in bytes).
    global MAX_VIRTUAL_MEMORY
    MAX_VIRTUAL_MEMORY = mem_limit * (1024 ** 3)

    status = "SUCCESS"
    runtime = 0
    try:
        io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE, preexec_fn=limit_virtual_memory)
        # Call target algorithm with cutoff time
        (stdout_, stderr_) = io.communicate(timeout=time_limit)
        if do_log:
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
            runtime = time_limit
    except TimeoutExpired:
        status = "TIMEOUT"
        runtime = time_limit

    return status, runtime
