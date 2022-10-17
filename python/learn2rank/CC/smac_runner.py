def main():
    fp = open('table.dat', 'w')
    idx = 1
    for si in range(1000, 1100, 10):
        fp.write(f'{idx} python -m learn2rank.scripts.smac_runner from_pid={si} split=val case={idx}\n')
        idx += 1


if __name__ == '__main__':
    main()
