case = 1


def create_table_line(case=0, problem='knapsack', n_objs=3, n_vars=60, split='train', order_type='canonical',
                      from_pid=0, to_pid=1):
    return f'{case} python -m learn2rank.scripts.eval_baseline_ordering ' \
           f'problem={problem} ' \
           f'problem.n_objs={n_objs} ' \
           f'problem.n_vars={n_vars} ' \
           f'order_type={order_type} ' \
           f'split={split} ' \
           f'from_pid={from_pid} ' \
           f'to_pid={to_pid} ' \
           f'machine=cc ' \
           f'case={case}\n'


def create_knapsack_table(splits=None):
    assert splits is not None

    global case
    size = [(3, 60), (3, 50), (3, 40), (3, 30), (3, 20),
            (4, 50), (4, 40), (4, 30), (4, 20),
            (5, 40), (5, 30), (5, 20)]
    order_type = ['min_weight', 'canonical']
    n_items = 50

    table_str = ''
    for ot in order_type:
        for split in splits:
            key, active, start, end = split
            if active:
                for s in size:
                    for si in range(start, end, n_items):
                        table_str += create_table_line(case=case, problem='knapsack', n_objs=s[0], n_vars=s[1],
                                                       split=key, from_pid=si, to_pid=si + n_items, order_type=ot)
                        case += 1

    return table_str


def create_setpacking_table(splits=None):
    assert splits is not None

    global case
    size = [(3, 100), (3, 125), (3, 150),
            (4, 100), (4, 125), (4, 150),
            (5, 100), (5, 125)]
    order_type = ['min_weight', 'canonical']
    n_items = 50

    table_str = ''
    for ot in order_type:
        for split in splits:
            key, active, start, end = split
            if active:
                for s in size:
                    for si in range(start, end, n_items):
                        table_str += create_table_line(case=case, problem='setpacking', n_objs=s[0], n_vars=s[1],
                                                       split=key, from_pid=si, to_pid=si + n_items, order_type=ot)
                        case += 1

    return table_str


def create_setcovering_table(splits=None):
    assert splits is not None

    global case
    size = [(3, 100), (3, 125),
            (4, 100), (4, 125),
            (5, 100)]
    order_type = ['min_weight', 'canonical']
    n_items = 50

    table_str = ''
    for ot in order_type:
        for split in splits:
            key, active, start, end = split
            if active:
                for s in size:
                    for si in range(start, end, n_items):
                        table_str += create_table_line(case=case, problem='setcovering', n_objs=s[0], n_vars=s[1],
                                                       split=key, from_pid=si, to_pid=si + n_items, order_type=ot)
                        case += 1

    return table_str


def main():
    gen_knapsack = True
    gen_setpacking = False
    gen_setcovering = False

    fp = open('table.dat', 'w')
    splits = (('train', True, 0, 1000),
              ('val', True, 1000, 1100),
              ('test', False, 1100, 1200))

    table_str = ''
    table_str = table_str + create_knapsack_table(splits=splits) if gen_knapsack else table_str
    table_str = table_str + create_setpacking_table(splits=splits) if gen_setpacking else table_str
    table_str = table_str + create_setcovering_table(splits=splits) if gen_setcovering else table_str

    fp.write(table_str)


if __name__ == '__main__':
    main()
