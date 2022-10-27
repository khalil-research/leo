def main():
    size = [(5, 40), (3, 60)]
    fp = open('table.dat', 'w')
    idx = 1
    cutoff = 60
    wallclock = 300
    n_instances = 25
    for s in size:
        for si in range(0, 1000, n_instances):
            fp.write(f'{idx} python -m learn2rank.scripts.smac_runner '
                     f'problem.n_objs={s[0]} '
                     f'problem.n_vars={s[1]} '
                     f'from_pid={si} '
                     f'num_instances={n_instances} '
                     f'cutoff_time={cutoff} '
                     f'wallclock_limit={wallclock} '
                     f'split=train '
                     f'case={idx}\n')
            idx += 1

        for si in range(1000, 1100, n_instances):
            fp.write(f'{idx} python -m learn2rank.scripts.smac_runner '
                     f'problem.n_objs={s[0]} '
                     f'problem.n_vars={s[1]} '
                     f'from_pid={si} '
                     f'num_instances={n_instances} '
                     f'cutoff_time={cutoff} '
                     f'wallclock_limit={wallclock} '
                     f'split=val '
                     f'case={idx}\n')
            idx += 1


if __name__ == '__main__':
    main()
