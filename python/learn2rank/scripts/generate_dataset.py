import json
import pickle as pkl
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from learn2rank.featurizer.factory import featurizer_factory
from learn2rank.utils.data import read_data_from_file
from learn2rank.utils.order import get_variable_rank_from_weights
from learn2rank.utils.order import property_weight_dict2array


@hydra.main(version_base='1.1', config_path='../config', config_name='generate_dataset.yaml')
def main(cfg: DictConfig):
    """Generate dataset

    Parameters
    ----------
    cfg

    Returns
    -------

    """
    res_path = Path(cfg.res_path[cfg.machine])
    dataset = {}

    # For each size
    for size in cfg.featurizer.size:
        if size not in dataset:
            dataset[size] = {}

        smac_out = res_path / 'smac_output' / cfg.featurizer.name / size
        for split in smac_out.iterdir():
            if split.stem not in dataset[size]:
                dataset[size][split.stem] = {}

            for inst in split.iterdir():
                dataset[size][split.stem][inst.stem] = {'x': {}, 'y': []}
                _dataset = dataset[size][split.stem][inst.stem]

                # Prepare x
                dat_path = res_path / 'instances' / cfg.featurizer.name / size / f'{split.stem}/{inst.stem}.dat'
                data = read_data_from_file(cfg.problem.acronym, dat_path)
                featurizer = featurizer_factory.create(cfg.featurizer.name, cfg=cfg.featurizer, data=data)
                features = featurizer.get()
                _dataset['x'] = features

                # Prepare y
                smac_out_dir = inst / f'run_{cfg.featurizer.seed.smac}'
                traj = smac_out_dir.joinpath('traj.json')
                with open(traj, 'r') as fp:
                    lines = fp.readlines()
                    for line in lines[1:]:
                        run = json.loads(line)
                        _dataset['y'].append({'cost': np.asarray(run['cost']).reshape(1, 1),
                                              'pwt': property_weight_dict2array(run['incumbent'],
                                                                                cast_to_numpy=True)})
                    _dataset['y'][-1]['rank'] = get_variable_rank_from_weights(
                        data, run['incumbent'], normalized=False)

    with open(res_path / f'datasets/{cfg.featurizer.name}/{cfg.featurizer.name}_dataset.pkl', 'wb') as fp:
        pkl.dump(dataset, fp)


if __name__ == '__main__':
    main()
