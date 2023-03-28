def main():
    idx = 1
    fp = open('table.dat', 'w')
    for perturb in ['random']:
        # 1 to 5
        for n_perturbations in [1]:
            # 0 to 50 with increments of 10
            for start_idx in [0]:
                for end_idx in [60]:
                    for pid in range(0, 500, 50):
                        for random_seed in [13, 444, 1212, 1003, 7517]:
                            fp.write(f'{idx} python -m learn2rank.scripts.perturbation_analysis '
                                     f'perturb.type={perturb} '
                                     f'perturb.times={n_perturbations} '
                                     f'perturb.start_idx={start_idx} '
                                     f'perturb.end_idx={end_idx} '
                                     f'from_pid={pid} '
                                     f'to_pid={pid + 50} '
                                     f'random_seed={random_seed} '
                                     f'case={idx}\n')
                            idx += 1

                    end_idx += 10


if __name__ == '__main__':
    main()
