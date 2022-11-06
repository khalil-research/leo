case = 1
size = [(3, 100)]
order_type = ['min_weight']
n_items = 50
run_train = True
run_val = False
problem = 'setpacking'

fp = open('table.dat', 'w')
for s in size:
    for ot in order_type:
        if run_train:
            for i in range(0, 50, n_items):
                fp.write(f'{case} python -m learn2rank.scripts.eval_baseline_ordering '
                         f'problem={problem} '
                         f'problem.n_objs={s[0]} '
                         f'problem.n_vars={s[1]} '
                         f'order_type={ot} '
                         f'from_pid={i} '
                         f'to_pid={i + n_items} '
                         f'split=train '
                         f'machine=cc '
                         f'case={case}\n')
                case += 1

        if run_val:
            for i in range(1000, 1100, n_items):
                fp.write(f'{case} python -m learn2rank.scripts.eval_baseline_ordering '
                         f'problem={problem} '
                         f'problem.n_objs={s[0]} '
                         f'problem.n_vars={s[1]} '
                         f'order_type={ot} '
                         f'from_pid={i} '
                         f'to_pid={i + n_items} '
                         f'split=val '
                         f'machine=cc '
                         f'case={case}\n')
                case += 1
