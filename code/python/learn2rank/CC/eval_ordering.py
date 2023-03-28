def main():
    idx = 1
    fp = open('table.dat', 'w')

    for split in ['val']:
        # 1 to 5
        for i in range(1000, 1100, 100):
            # 0 to 50 with increments of 10
            j = i + 100
            if j <= 1000:
                fp.write(f'{idx} python -m learn2rank.scripts.eval_ordering '
                         f'from_pid={i} '
                         f'to_pid={j} '
                         f'split={split} '
                         f'random.switch=1 '
                         f'case={idx}\n')
                idx += 1


if __name__ == '__main__':
    main()
