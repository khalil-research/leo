import json
import pickle as pkl
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from learn2rank.featurizer.factory import featurizer_factory
from learn2rank.utils import get_variable_rank_from_weights
from learn2rank.utils import property_weight_dict2array
from learn2rank.utils import read_data_from_file


@hydra.main(version_base='1.1', config_path='../config', config_name='generate_dataset.yaml')
def main(cfg: DictConfig):
    """Generate dataset

    Parameters
    ----------
    cfg

    Returns
    -------

    """
    cfg = cfg.featurizer

    dp = Path(cfg.paths.data)
    assert dp.exists(), "Invalid dataset path!"
    dataset = {}

    # For each data folder
    for df in dp.iterdir():
        of = Path(cfg.paths.output).joinpath(df.name)
        if not of.exists():
            continue

        # For each split
        for dfs in df.iterdir():
            # Add split key
            if dfs.name not in dataset:
                dataset[dfs.name] = {}

            for data_file in dfs.iterdir():
                data = read_data_from_file(data_file)
                dataset[dfs.name][data_file.stem] = {'x': {}, 'y': []}
                _dataset = dataset[dfs.name][data_file.stem]

                featurizer = featurizer_factory.create(cfg.name, cfg=cfg, data=data)
                features = featurizer.get()
                _dataset['x'] = features

                # Load incumbent config
                smac_out_dir = of.joinpath(dfs.name).joinpath(data_file.stem).joinpath(f'run_{cfg.seed.smac}')
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

    with open(Path(cfg.paths.dataset) / 'knapsack.pkl', 'wb') as fp:
        pkl.dump(dataset, fp)


if __name__ == '__main__':
    main()
