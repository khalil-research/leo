from pathlib import Path

import numpy as np

from learn2rank.utils.data import read_data_from_file

sizes = ['100_3', '100_4', '100_5', '100_6', '100_7',
         '150_3', '150_4']
CURR_FILE_PATH = Path(__file__)
RES_PATH = CURR_FILE_PATH.parent.parent / 'resources'
BIN_PATH = RES_PATH / 'instances/binproblem'

SETCOVER_PATH = RES_PATH / 'instances/setcovering'


def write_to_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    n_objs = data['value'].shape[0]
    n_vars = data['value'][0].shape[0]
    n_cons = data['cons_mat'].shape[0]

    text = f'{n_vars} {n_cons}\n{n_objs}\n'
    for i in range(n_objs):
        string = " ".join([str(v) for v in data['value'][i]])
        text += string + "\n"

    for i in range(n_cons):
        text += f"{len(data['cons'][i])} \n"
        string = " ".join([str(c) for c in data['cons'][i]])
        text += string + "\n"

    path.open('w').write(text)


for size in sizes:

    split_path = BIN_PATH / size / 'train'
    for i in range(1000):
        data = read_data_from_file('bp', split_path / f'bp_7_{size}_{i}.dat')

        data['cons'] = []
        for j in range(data['cons_mat'].shape[0]):
            idxs = np.arange(data['n_vars'])
            ids = idxs[data['cons_mat'][j] == 1]
            ids += 1

            data['cons'].append(list(ids))

        write_to_file(SETCOVER_PATH / size / 'train' / f'bp_7_{size}_{i}.dat', data)

    split_path = BIN_PATH / size / 'val'
    for i in range(1000, 1100):
        data = read_data_from_file('bp', split_path / f'bp_7_{size}_{i}.dat')

        data['cons'] = []
        for j in range(data['cons_mat'].shape[0]):
            idxs = np.arange(data['n_vars'])
            ids = idxs[data['cons_mat'][j] == 1]
            ids += 1

            data['cons'].append(list(ids))

        write_to_file(SETCOVER_PATH / size / 'val' / f'bp_7_{size}_{i}.dat', data)

    split_path = BIN_PATH / size / 'test'
    for i in range(1100, 1200):
        data = read_data_from_file('bp', split_path / f'bp_7_{size}_{i}.dat')

        data['cons'] = []
        for j in range(data['cons_mat'].shape[0]):
            idxs = np.arange(data['n_vars'])
            ids = idxs[data['cons_mat'][j] == 1]
            ids += 1

            data['cons'].append(list(ids))

        write_to_file(SETCOVER_PATH / size / 'test' / f'bp_7_{size}_{i}.dat', data)
