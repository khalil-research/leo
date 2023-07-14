import hydra
import numpy as np
from omegaconf import DictConfig

from leo import path


def generate_instance(rng, n_vars, n_objs, max_obj=1000):
    data = {'value': [], 'weight': [], 'capacity': 0}

    # Value
    for _ in range(n_objs):
        data['value'].append(rng.randint(1, max_obj + 1, n_vars))
        # Cost
    data['weight'] = rng.randint(1, max_obj + 1, n_vars)
    # Capacity
    data['capacity'] = np.ceil(0.5 * (np.sum(data['weight'])))

    return data


def write_to_file(inst_path, data):
    inst_path.parent.mkdir(parents=True, exist_ok=True)

    n_vars = len(list(data['weight']))
    n_objs = len(data['value'])

    text = f'{n_vars}\n{n_objs}\n'
    for i in range(n_objs):
        string = ' '.join([str(v) for v in data['value'][i]])
        text += string + '\n'
    string = ' '.join([str(w) for w in data['weight']])
    text += string + '\n'
    text += str(int(data['capacity']))

    inst_path.open('w').write(text)


@hydra.main(version_base='1.2', config_path='./config', config_name='generate_instance.yaml')
def main(cfg: DictConfig):
    rng = np.random.RandomState(cfg.seed)

    for s in cfg.size:
        n_objs, n_vars = map(int, s.split('_'))

        for id in range(cfg.n_train):
            write_to_file(
                path.instances / f'{cfg.name}/{n_objs}_{n_vars}/train/kp_{cfg.seed}_{n_objs}_{n_vars}_{id}.dat',
                generate_instance(rng, n_vars, n_objs, max_obj=cfg.max_obj))

        start = cfg.n_train
        end = start + cfg.n_val
        for id in range(start, end):
            write_to_file(
                path.instances / f'{cfg.name}/{n_objs}_{n_vars}/val/kp_{cfg.seed}_{n_objs}_{n_vars}_{id}.dat',
                generate_instance(rng, n_vars, n_objs, max_obj=cfg.max_obj))

        start = cfg.n_train + cfg.n_val
        end = start + cfg.n_test
        for id in range(start, end):
            write_to_file(
                path.instances / f'{cfg.name}/{n_objs}_{n_vars}/test/kp_{cfg.seed}_{n_objs}_{n_vars}_{id}.dat',
                generate_instance(rng, n_vars, n_objs, max_obj=cfg.max_obj))


if __name__ == '__main__':
    main()
