from collections import namedtuple

Config = namedtuple('Config', ['cutoff_time', 'wallclock_limit', 'n_instances'])

case = 1

# Modes
SMAC_ALL = 'all'
SMAC_ONE = 'one'

# In hours
HOUR2SEC = 60 * 60
JOB_TIME = 3 * HOUR2SEC

# Cutoff presets
BASE_CUTOFF = 60
BASE_CUTOFF_2 = 2 * BASE_CUTOFF
BASE_CUTOFF_3 = 3 * BASE_CUTOFF
BASE_CUTOFF_4 = 4 * BASE_CUTOFF
BASE_CUTOFF_ALL = 10 * BASE_CUTOFF

# Wallclock presets
BASE_WALLCLOCK = 5 * 60
BASE_WALLCLOCK_2 = 2 * BASE_WALLCLOCK
BASE_WALLCLOCK_3 = 3 * BASE_WALLCLOCK
BASE_WALLCLOCK_4 = 4 * BASE_WALLCLOCK
BASE_WALLCLOCK_ALL = 12 * HOUR2SEC

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


def create_table_line(case=0, problem='knapsack', n_objs=3, n_vars=60, split='train', start_idx=0, n_instances=1,
                      cutoff=60, wallclock=300, init_incumbent='canonical', mode='one'):
    return f'{case} python -m learn2rank.scripts.smac_runner ' \
           f'problem={problem} ' \
           f'problem.n_objs={n_objs} ' \
           f'problem.n_vars={n_vars} ' \
           f'split={split} ' \
           f'from_pid={start_idx} ' \
           f'num_instances={n_instances} ' \
           f'cutoff_time={cutoff} ' \
           f'wallclock_limit={wallclock} ' \
           f'init_incumbent={init_incumbent} ' \
           f'mode={mode} ' \
           f'machine=cc ' \
           f'case={case}\n'


def create_knapsack_table(splits=None, mode='one'):
    assert splits is not None

    global case
    configs_one = {
        '40_5': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 2),
        '50_4': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 2),
        '60_3': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 2),
        '40_6': Config(BASE_CUTOFF_2, BASE_WALLCLOCK_2, int(JOB_TIME / BASE_WALLCLOCK_2) - 2),
        '70_3': Config(BASE_CUTOFF_2, BASE_WALLCLOCK_2, int(JOB_TIME / BASE_WALLCLOCK_2) - 2),
        '40_7': Config(BASE_CUTOFF_4, BASE_WALLCLOCK_4, int(JOB_TIME / BASE_WALLCLOCK_4) - 1),
        '80_3': Config(BASE_CUTOFF_4, BASE_WALLCLOCK_4, int(JOB_TIME / BASE_WALLCLOCK_4) - 1)
    }
    size = [(3, 60), (3, 70), (3, 80),
            (4, 50),
            (5, 40),
            (6, 40),
            (7, 40)]

    table_str = ''
    for split in splits:
        key, active, start, end = split
        if active:
            for s in size:
                _cfg = configs_one[f'{s[1]}_{s[0]}'] if mode == 'one' else configs_all[f'{s[1]}_{s[0]}']

                for si in range(start, end, _cfg.n_instances):
                    _n_instances = _cfg.n_instances if si + _cfg.n_instances < end else end - si
                    table_str += create_table_line(case=case, problem='knapsack', n_objs=s[0], n_vars=s[1], split=key,
                                                   start_idx=si, n_instances=_n_instances,
                                                   cutoff=_cfg.cutoff_time, wallclock=_cfg.wallclock_limit,
                                                   init_incumbent='min_weight', mode=mode)
                    case += 1

    return table_str


def create_setpacking_table(splits=None, mode='one'):
    assert splits is not None

    global case
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


def create_setcovering_table(splits=None, mode='one'):
    assert splits is not None

    global case

    configs_one = {
        '100_3': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 2),
        '100_4': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 2),
        '100_5': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 2),
        '100_6': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 2),
        '100_7': Config(BASE_CUTOFF, BASE_WALLCLOCK, int(JOB_TIME / BASE_WALLCLOCK) - 2),
        '150_3': Config(BASE_CUTOFF_2, BASE_WALLCLOCK_2, int(JOB_TIME / BASE_WALLCLOCK_2) - 1),
        '150_4': Config(BASE_CUTOFF_4, BASE_WALLCLOCK_4, int(JOB_TIME / BASE_WALLCLOCK_4) - 1)
    }
    size = [(3, 100), (4, 100), (5, 100), (6, 100), (7, 100),
            (3, 150), (4, 150)]

    table_str = ''
    for split in splits:
        key, active, start, end = split
        if active:
            for s in size:
                _cfg = configs_one[f'{s[1]}_{s[0]}'] if mode == 'one' else configs_all[f'{s[1]}_{s[0]}']

                for si in range(start, end, _cfg.n_instances):
                    _n_instances = _cfg.n_instances if si + _cfg.n_instances < end else end - si
                    table_str += create_table_line(case=case, problem='setcovering', n_objs=s[0], n_vars=s[1],
                                                   split=key, start_idx=si, n_instances=_n_instances,
                                                   cutoff=_cfg.cutoff_time, wallclock=_cfg.wallclock_limit,
                                                   init_incumbent='max_weight', mode=mode)
                    case += 1

    return table_str


def main():
    gen_knapsack = True
    gen_setpacking = True
    gen_setcovering = True

    fp = open('table.dat', 'w')
    splits = (('train', True, 0, 1000),
              ('val', True, 1000, 1100),
              ('test', False, 1100, 1200))
    # Can be SMAC_ONE | SMAC_ALL
    mode = SMAC_ONE

    table_str = ''
    table_str = table_str + create_knapsack_table(splits=splits, mode=mode) if gen_knapsack else table_str
    table_str = table_str + create_setcovering_table(splits=splits, mode=mode) if gen_setcovering else table_str
    table_str = table_str + create_setpacking_table(splits=splits, mode=mode) if gen_setpacking else table_str

    fp.write(table_str)


if __name__ == '__main__':
    main()
