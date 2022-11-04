case = 1
size = [(3, 60), (5, 40)]
order_type = ['rand']
n_items = 50

fp = open('table.dat', 'w')
for s in size:
    for ot in order_type:
        for i in range(0, 1000, n_items):
            fp.write(f'{case} python -m learn2rank.scripts.eval_static_orderings '
                     f'problem.n_objs={s[0]} '
                     f'problem.n_vars={s[1]} '
                     f'order_type={ot} '
                     f'from_pid={i} '
                     f'to_pid={i + n_items} '
                     f'split=train '
                     f'case={case}\n')
            case += 1

        for i in range(1000, 1100, n_items):
            fp.write(f'{case} python -m learn2rank.scripts.eval_static_orderings '
                     f'problem.n_objs={s[0]} '
                     f'problem.n_vars={s[1]} '
                     f'order_type={ot} '
                     f'from_pid={i} '
                     f'to_pid={i + n_items} '
                     f'split=val '
                     f'case={case}\n')
            case += 1
