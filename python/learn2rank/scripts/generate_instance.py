from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig


def generate_knapsack_instances(cfg):
    def generate_instance(n_vars, n_objs, max_obj=1000):
        data = {'value': [], 'weight': [], 'capacity': 0}

        # Value
        for _ in range(n_objs):
            data['value'].append(rng.randint(1, max_obj + 1, n_vars))
            # Cost
        data['weight'] = rng.randint(1, max_obj + 1, n_vars)
        # Capacity
        data['capacity'] = np.ceil(0.5 * (np.sum(data['weight'])))

        return data

    def write_to_file(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)

        n_vars = len(list(data['weight']))
        n_objs = len(data['value'])

        text = f'{n_vars}\n{n_objs}\n'
        for i in range(n_objs):
            string = " ".join([str(v) for v in data['value'][i]])
            text += string + "\n"
        string = " ".join([str(w) for w in data['weight']])
        text += string + "\n"
        text += str(int(data['capacity']))

        path.open('w').write(text)

    rng = np.random.RandomState(cfg.seed)
    for s in cfg.size:
        n_vars, n_objs = map(int, s.split('-'))

        for id in range(cfg.n_train):
            write_to_file(
                Path(f'./mo_instances/{cfg.name}/{n_objs}_{n_vars}/train/kp_{cfg.seed}_{n_objs}_{n_vars}_{id}.dat'),
                generate_instance(n_vars, n_objs, max_obj=cfg.max_obj))

        start = cfg.n_train
        end = start + cfg.n_val
        for id in range(start, end):
            write_to_file(
                Path(f'./mo_instances/{cfg.name}/{n_objs}_{n_vars}/val/kp_{cfg.seed}_{n_objs}_{n_vars}_{id}.dat'),
                generate_instance(n_vars, n_objs, max_obj=cfg.max_obj))

        start = cfg.n_train + cfg.n_val
        end = start + cfg.n_test
        for id in range(start, end):
            write_to_file(
                Path(f'{cfg.output_dir}/{cfg.name}/{n_objs}_{n_vars}/test/kp_{cfg.seed}_{n_objs}_{n_vars}_{id}.dat'),
                generate_instance(n_vars, n_objs, max_obj=cfg.max_obj))


def generate_binproblem_instances(cfg):
    def write_to_file(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)

        n_objs = len(data['value'])
        n_vars = len(data['value'][0])
        n_cons = int(n_vars / 5)

        text = f'{n_vars} {n_cons}\n{n_objs}\n'
        for i in range(n_objs):
            string = " ".join([str(v) for v in data['value'][i]])
            text += string + "\n"

        for i in range(n_cons):
            text += f"{len(data['cons'])} \n"
            string = " ".join([str(c) for c in data['cons'][i]])
            text += string + "\n"

        path.open('w').write(text)

    def generate_instance(n_vars, n_objs, max_obj=100):
        n_cons = int(n_vars / 5)

        # Fixed
        # 2 to 18 constraints per variable
        n_vars_per_con = rng.randint(2, 19)

        data = {'value': [], 'cons': []}
        for _ in range(n_objs):
            data['value'].append(rng.randint(1, max_obj + 1, n_vars))

        for _ in range(n_cons):
            data['cons'].append(rng.choice(n_vars, n_vars_per_con, replace=False) + 1)

        return data

    rng = np.random.RandomState(cfg.seed)
    for s in cfg.size:
        n_vars, n_objs = map(int, s.split('-'))

        for id in range(cfg.n_train):
            write_to_file(
                Path(f'./mo_instances/{cfg.name}/{n_vars}_{n_objs}/train/bp_{cfg.seed}_{n_vars}_{n_objs}_{id}.dat'),
                generate_instance(n_vars, n_objs, max_obj=cfg.max_obj))

        start = cfg.n_train
        end = start + cfg.n_val
        for id in range(start, end):
            write_to_file(
                Path(f'./mo_instances/{cfg.name}/{n_vars}_{n_objs}/val/bp_{cfg.seed}_{n_vars}_{n_objs}_{id}.dat'),
                generate_instance(n_vars, n_objs, max_obj=cfg.max_obj))

        start = cfg.n_train + cfg.n_val
        end = start + cfg.n_test
        for id in range(start, end):
            write_to_file(
                Path(f'{cfg.output_dir}/{cfg.name}/{n_vars}_{n_objs}/test/bp_{cfg.seed}_{n_vars}_{n_objs}_{id}.dat'),
                generate_instance(n_vars, n_objs, max_obj=cfg.max_obj))


@hydra.main(version_base='1.2', config_path='../config', config_name='generate_instance.yaml')
def main(cfg: DictConfig):
    if cfg.knapsack.generate:
        generate_knapsack_instances(cfg.knapsack)

    if cfg.binproblem.generate:
        generate_binproblem_instances(cfg.binproblem)


if __name__ == '__main__':
    main()
