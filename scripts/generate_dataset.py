import numpy as np

from operator import itemgetter
from pathlib import Path
from argparse import ArgumentParser


def write_to_file(path, p, n, data):
    text = f'{n}\n{p}\n'
    for i in range(p):
        string = " ".join([str(v) for v in data['value'][i]])
        text += string + "\n"
    string = " ".join([str(w) for w in data['weight']])
    text += string + "\n"
    text += str(int(data['capacity']))

    path.open('w').write(text)


def generate_dataset(obj_var_cfg, seed=7, n_instances=10000, n_train=8000, n_val=1000, n_test=1000, root='dataset'):
    assert n_instances == n_train + n_val + n_test
    np.random.seed(seed)
    root = Path(root)
    root = root / f'knapsack_{seed}'
    root.mkdir(parents=True, exist_ok=True)

    for p in obj_var_cfg.keys():
        for n in obj_var_cfg[p]:
            prob_root = root / f'{p}_{n}'
            prob_root.mkdir(parents=True, exist_ok=True)

            dataset = []
            for i in range(n_instances):
                data = {'value': [], 'weight': [], 'capacity': 0}
                # Value
                for _ in range(p):
                    data['value'].append(np.random.randint(1, 1001, n))
                # Cost
                data['weight'] = np.random.randint(1, 1001, n)
                # Capacity
                data['capacity'] = np.ceil(0.5 * (np.sum(data['weight'])))
                dataset.append(data)

            set_root = prob_root / 'train'
            set_root.mkdir(parents=True, exist_ok=True)
            for i in range(n_train):
                inst_root = set_root / f'kp_{seed}_{p}_{n}_{i}.dat'
                write_to_file(inst_root, p, n, dataset[i])

            set_root = prob_root / 'val'
            set_root.mkdir(parents=True, exist_ok=True)
            for i in range(n_train, n_train + n_val):
                inst_root = set_root / f'kp_{seed}_{p}_{n}_{i}.dat'
                write_to_file(inst_root, p, n, dataset[i])

            set_root = prob_root / 'test'
            set_root.mkdir(parents=True, exist_ok=True)
            for i in range(n_train + n_val, n_train + n_val + n_test):
                inst_root = set_root / f'kp_{seed}_{p}_{n}_{i}.dat'
                write_to_file(inst_root, p, n, dataset[i])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--n_instances', type=int, default=10000)
    parser.add_argument('--n_train', type=int, default=8000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=1000)
    args = parser.parse_args()

    obj_var_cfg = {3: [20, 40, 60],
                   5: [20, 40],
                   7: [20, 30]}

    generate_dataset(obj_var_cfg, seed=args.seed, n_instances=args.n_instances,
                     n_test=args.n_test, n_val=args.n_val, n_train=args.n_train)
