import ast
import pickle as pkl
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from learn2rank.featurizer.factory import featurizer_factory
from learn2rank.utils.data import read_data_from_file
from learn2rank.utils.order import get_variable_rank_from_weights


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
    inst_root_path = res_path / 'instances' / cfg.problem.name
    dataset_path = res_path / 'datasets' / cfg.problem.name / f'{cfg.problem.name}_dataset.pkl'
    dataset = {} if not dataset_path.exists() else pkl.load(open(dataset_path, 'rb'))

    # For each size
    for size in cfg.featurizer.size:
        if size not in dataset:
            dataset[size] = {}

        df_label = pd.read_csv(res_path / 'raw_labels' / cfg.problem.name / f'label_{size}.csv')
        print(df_label.shape)
        for split in ['train']:
            if split not in dataset[size]:
                dataset[size][split] = {}

            inst_path = inst_root_path / size / split
            for inst in inst_path.iterdir():
                dataset[size][split][inst.stem] = {'x': {}, 'y': []}
                _dataset = dataset[size][split][inst.stem]

                # Prepare x
                data = read_data_from_file(cfg.problem.acronym, inst)
                featurizer = featurizer_factory.create(cfg.featurizer.name, cfg=cfg.featurizer, data=data)
                features = featurizer.get()
                _dataset['x'] = features

                pid = int(inst.stem.split('.')[0].split('_')[-1])
                print(inst, pid)
                label_row = df_label[df_label['pid'] == pid]
                print(label_row)
                print(label_row['incb'].values[0], type(label_row['incb'].values[0]))

                incb_dict = ast.literal_eval(label_row['incb'].values[0])
                _dataset['y'] = get_variable_rank_from_weights(data, incb_dict, normalized=False)

    with open(res_path / f'datasets/{cfg.problem.name}/{cfg.problem.name}_dataset.pkl', 'wb') as fp:
        pkl.dump(dataset, fp)


if __name__ == '__main__':
    main()
