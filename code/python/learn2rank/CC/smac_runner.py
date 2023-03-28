from collections import namedtuple

Config = namedtuple('Config', ['cutoff_time', 'wallclock_limit', 'n_instances'])

case = 1

# Modes
smacD = 'all'
smacI = 'one'

# In hours
HOUR2SEC = 60 * 60
JOB_TIME = 3 * HOUR2SEC

# Cutoff and Wallclock presets
# We set cutoff equal to wallclock to avoid early termination
BASE_CUTOFF = BASE_WALLCLOCK = 5 * 60
BASE_CUTOFF_2 = BASE_WALLCLOCK_2 = 2 * BASE_WALLCLOCK
BASE_CUTOFF_3 = BASE_WALLCLOCK_3 = 3 * BASE_WALLCLOCK
BASE_CUTOFF_4 = BASE_WALLCLOCK_4 = 4 * BASE_WALLCLOCK
BASE_CUTOFF_ALL = BASE_WALLCLOCK_ALL = 12 * HOUR2SEC

configs_all = {
    '40_5': Config(BASE_CUTOFF_ALL, BASE_WALLCLOCK_ALL, 1000),
    '50_4': Config(BASE_CUTOFF_ALL, BASE_WALLCLOCK_ALL, 1000),
    '60_3': Config(BASE_CUTOFF_ALL, BASE_WALLCLOCK_ALL, 1000),
    '40_6': Config(BASE_CUTOFF_ALL, BASE_WALLCLOCK_ALL, 1000),
    '70_3': Config(BASE_CUTOFF_ALL, BASE_WALLCLOCK_ALL, 1000),
    '40_7': Config(BASE_CUTOFF_ALL, BASE_WALLCLOCK_ALL, 1000),
    '80_3': Config(BASE_CUTOFF_ALL, BASE_WALLCLOCK_ALL, 1000),
    '100_3': Config(BASE_CUTOFF_ALL, BASE_WALLCLOCK_ALL, 1000),
    '100_4': Config(BASE_CUTOFF_ALL, BASE_WALLCLOCK_ALL, 1000),
    '100_5': Config(BASE_CUTOFF_ALL, BASE_WALLCLOCK_ALL, 1000),
    '100_6': Config(BASE_CUTOFF_ALL, BASE_WALLCLOCK_ALL, 1000),
    '100_7': Config(BASE_CUTOFF_ALL, BASE_WALLCLOCK_ALL, 1000),
    '150_3': Config(BASE_CUTOFF_ALL, BASE_WALLCLOCK_ALL, 1000),
    '150_4': Config(BASE_CUTOFF_ALL, BASE_WALLCLOCK_ALL, 1000),
    '150_5': Config(BASE_CUTOFF_ALL, BASE_WALLCLOCK_ALL, 1000)
}

# seeds = [329, 1021, 983439, 4321, 7623, 5621, 271, 82, 3336, 813,
#          774, 9194, 2127, 5104, 7746, 2401, 76, 5475, 7557, 5958, 6348,
#          7766, 7843, 533, 8496, 1234, 7186, 7987, 1245, 2959, 470, 8946,
#          571, 6654, 9314, 7144, 7179, 3198, 854, 7774, 5953, 4226, 2857,
#          3345, 578, 2020, 1253, 1337, 8695, 8385]

seeds = [578, 470, 1337, 3345, 983439, 329, 1021, 4321, 7623, 5621,
         271, 82, 3336, 813, 774, 9194, 2127, 5104, 7746, 2401, 76,
         5475, 7557, 5958, 6348, 7766, 7843, 533, 8496, 1234, 7186,
         7987, 1245, 2959, 8946, 571, 6654, 9314, 7144, 7179, 3198,
         854, 7774, 5953, 4226, 2857, 2020, 1253, 8695, 8385]


def create_table_line(case=0, problem='knapsack', n_objs=3, n_vars=60, bin_name='multiobj', mode='one', seed=777,
                      n_jobs=1, cutoff=60, wallclock=300, init_incumbent='canonical', restore_run=0, new_cutoff=120,
                      new_wallclock=600, mask_mem_limit=0, mem_limit=16, default_width=1.0, label_width=1.0,
                      split='train', start_idx=0, n_instances=1):
    return f'{case} python -m learn2rank.smac_runner ' \
           f'problem={problem} ' \
           f'problem.n_objs={n_objs} ' \
           f'problem.n_vars={n_vars} ' \
           f'bin_name={bin_name} ' \
           f'mode={mode} ' \
           f'seed={seed} ' \
           f'n_jobs={n_jobs} ' \
           f'cutoff_time={cutoff} ' \
           f'wallclock_limit={wallclock} ' \
           f'init_incumbent={init_incumbent} ' \
           f'restore_run={restore_run} ' \
           f'new_cutoff_time={new_cutoff} ' \
           f'new_wallclock_limit={new_wallclock} ' \
           f'mask_mem_limit={mask_mem_limit} ' \
           f'mem_limit={mem_limit} ' \
           f'width.default={default_width} ' \
           f'width.label={label_width} ' \
           f'split={split} ' \
           f'from_pid={start_idx} ' \
           f'num_instances={n_instances} ' \
           f'machine=cc ' \
           f'case={case}\n'


def create_knapsack_table():
    global case
    configs_one = {
        '40_5': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 1),
        '50_4': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 1),
        '60_3': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 1),
        '40_6': Config(BASE_CUTOFF_4, BASE_WALLCLOCK_4, int(JOB_TIME / BASE_WALLCLOCK_4) - 1),
        '70_3': Config(BASE_CUTOFF_4, BASE_WALLCLOCK_4, int(JOB_TIME / BASE_WALLCLOCK_4) - 1),
        '40_7': Config(BASE_CUTOFF_4, BASE_WALLCLOCK_4, int(JOB_TIME / BASE_WALLCLOCK_4) - 1),
        '80_3': Config(BASE_CUTOFF_4, BASE_WALLCLOCK_4, int(JOB_TIME / BASE_WALLCLOCK_4) - 1)
    }
    # size = [(3, 60), (3, 70), (3, 80),
    #         (4, 50),
    #         (5, 40),
    #         (6, 40),
    #         (7, 40)]
    size = [(4, 50)]
    table_str = ''
    # seed = 777
    for split in splits:
        key, active, start, end = split
        if not active:
            continue

        for s in size:
            _cfg = configs_one[f'{s[1]}_{s[0]}'] if mode == smacI else configs_all[f'{s[1]}_{s[0]}']
            for seed_id in range(seeds_start_idx, seeds_start_idx + n_seeds):
                # NOTE: Seeds play an important role in the performance for size (3, 70)
                # Also, setting the cutoff limit higher helps.
                for si in range(start, end, _cfg.n_instances):
                    _n_instances = _cfg.n_instances if si + _cfg.n_instances < end else end - si
                    table_str += create_table_line(case=case, problem='knapsack', n_objs=s[0], n_vars=s[1],
                                                   split=key,
                                                   start_idx=si, n_instances=_n_instances,
                                                   cutoff=_cfg.cutoff_time, wallclock=_cfg.wallclock_limit,
                                                   restore_run=restore_run, new_wallclock=new_wallclock,
                                                   new_cutoff=new_cutoff,
                                                   init_incumbent=init_incumbent, mode=mode,
                                                   n_jobs=n_jobs,
                                                   seed=seeds[seed_id])
                    case += 1

    return table_str


def create_setpacking_table():
    # global case
    configs_one = {
        '150_3': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 2),
        '150_4': Config(BASE_CUTOFF_2, BASE_WALLCLOCK_2, int(JOB_TIME / BASE_WALLCLOCK_2) - 2),
        '150_5': Config(BASE_CUTOFF_4, BASE_WALLCLOCK_4, int(JOB_TIME / BASE_WALLCLOCK_4) - 1)
    }
    size = [(3, 150), (4, 150), (5, 150)]

    table_str = ''
    for split in splits:
        key, active, start, end = split
        if active:
            for s in size:
                _cfg = configs_one[f'{s[1]}_{s[0]}'] if mode == 'one' else configs_all[f'{s[1]}_{s[0]}']

                for si in range(start, end, _cfg.n_instances):
                    _n_instances = _cfg.n_instances if si + _cfg.n_instances < end else end - si
                    table_str += create_table_line(case=case, problem='setpacking', n_objs=s[0], n_vars=s[1], split=key,
                                                   start_idx=si, n_instances=_n_instances,
                                                   cutoff=_cfg.cutoff_time, wallclock=_cfg.wallclock_limit,
                                                   init_incumbent='canonical', mode=mode)
                    case += 1

    return table_str


def create_setcovering_table():
    global case

    configs_one = {
        '100_3': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 2),
        '100_4': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 2),
        '100_5': Config(BASE_CUTOFF, BASE_WALLCLOCK_2, int(JOB_TIME / BASE_WALLCLOCK_2) - 1),
        '100_6': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 2),
        '100_7': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 2),
        '150_3': Config(BASE_CUTOFF_2, BASE_WALLCLOCK_2, int(JOB_TIME / BASE_WALLCLOCK_2) - 1),
        '150_4': Config(BASE_CUTOFF_4, BASE_WALLCLOCK_4, int(JOB_TIME / BASE_WALLCLOCK_4) - 1)
    }
    # size = [(3, 150), (4, 150),
    #         (3, 100), (4, 100), (5, 100), (6, 100), (7, 100)]
    size = [(5, 100)]

    table_str = ''
    for split in splits:
        key, active, start, end = split
        if active:
            for s in size:
                _cfg = configs_one[f'{s[1]}_{s[0]}'] if mode == 'one' else configs_all[f'{s[1]}_{s[0]}']
                for seed_id in range(seeds_start_idx, seeds_start_idx + n_seeds):
                    for si in range(start, end, _cfg.n_instances):
                        for dw in [0.25, 0.5, 0.75, 1]:
                            for lw in [0.25, 0.5, 0.75, 1]:
                                _n_instances = _cfg.n_instances if si + _cfg.n_instances < end else end - si
                                table_str += create_table_line(case=case, problem='setcovering', n_objs=s[0],
                                                               n_vars=s[1],
                                                               split=key, start_idx=si, n_instances=_n_instances,
                                                               cutoff=_cfg.cutoff_time, wallclock=_cfg.wallclock_limit,
                                                               mask_mem_limit=mask_mem_limit, mem_limit=mem_limit,
                                                               default_width=dw, label_width=lw,
                                                               init_incumbent='canonical',
                                                               mode=mode, seed=seeds[seed_id])
                                case += 1

    return table_str


# Select problem
gen_knapsack = True
gen_setpacking = False
gen_setcovering = False
# Select split and problem ids
splits = (('train', False, 0, 1000),
          ('val', True, 1000, 1100),
          ('test', True, 1100, 1200))
# Select number of seeds you want to try
seeds_start_idx = 0
n_seeds = 5
# Select mode
mode = smacI
# Select number of jobs
n_jobs = 1
# Select initial incumbent
init_incumbent = 'min_weight'
# Restore previous run
restore_run = 0
new_cutoff = 600
new_wallclock = 2400

mask_mem_limit = 0
mem_limit = 16

default_width = 1
label_width = 1


def main():
    fp = open('table.dat', 'w')

    table_str = ''
    table_str = table_str + create_knapsack_table() if gen_knapsack else table_str
    table_str = table_str + create_setcovering_table() if gen_setcovering else table_str
    table_str = table_str + create_setpacking_table() if gen_setpacking else table_str

    fp.write(table_str)


if __name__ == '__main__':
    main()
