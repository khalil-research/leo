import ast
import json
import pickle as pkl
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from learn2rank.featurizer.factory import featurizer_factory
from learn2rank.utils.data import read_data_from_file
from learn2rank.utils.order import get_variable_rank_from_weights
from learn2rank.utils.order import property_weight_dict2array


def generate_dataset_point_regress(cfg):
    res_path = Path(cfg.res_path[cfg.machine])
    inst_root_path = res_path / 'instances' / cfg.problem.name
    dataset_path = res_path / 'datasets' / cfg.problem.name / f'{cfg.problem.name}_dataset_{cfg.task}.pkl'
    dataset = {} if not dataset_path.exists() else pkl.load(open(dataset_path, 'rb'))
    time_dataset = []

    # For each size
    for size in cfg.size:
        if size in dataset:
            print("Overwriting dataset for size ", size, "... \n")
        dataset[size] = {}

        for split in cfg.split:
            dataset[size][split] = []
            df_label = pd.read_csv(res_path / 'labels' / cfg.problem.name / size / f'label_{size}_{split}.csv')
            print(f"Split {split}, labels shape {df_label.shape}")

            inst_path = inst_root_path / size / split
            for inst in inst_path.iterdir():
                pid = int(inst.stem.split('.')[0].split('_')[-1])
                sample = {'name': inst.stem, 'pid': pid, 'seed': None, 'x': {}, 'y': []}

                start_time = time.time()
                # Prepare x
                data = read_data_from_file(cfg.problem.acronym, inst)
                featurizer = featurizer_factory.create(cfg.featurizer.name, cfg=cfg.featurizer, data=data)
                features = featurizer.get()
                sample['x'] = features

                # Prepare y
                # Get best seed for the pid
                label_row = df_label[df_label['pid'] == pid]
                best_seed = label_row['seed'].values[0]
                sample['seed'] = best_seed
                # Save the incumbent
                incb_dict = ast.literal_eval(label_row['incb'].values[0])
                sample['y'] = get_variable_rank_from_weights(data, incb_dict, normalized=bool(cfg.normalize_rank))

                end_time = time.time() - start_time
                time_dataset.append([size, pid, best_seed, split, end_time])

                # Append sample to dataset
                dataset[size][split].append(sample)

    # Save time
    time_df = pd.DataFrame(time_dataset, columns=["size", "pid", "best_seed", "split", "time"])
    time_df.to_csv(res_path / f'datasets/{cfg.problem.name}/{cfg.problem.name}_time_dataset_{cfg.task}.csv',
                   index=False)

    # Save dataset
    with open(res_path / f'datasets/{cfg.problem.name}/{cfg.problem.name}_dataset_{cfg.task}.pkl', 'wb') as fp:
        pkl.dump(dataset, fp)


def generate_dataset_multitask(cfg):
    res_path = Path(cfg.res_path[cfg.machine])
    inst_root_path = res_path / 'instances' / cfg.problem.name
    dataset_path = res_path / 'datasets' / cfg.problem.name / f'{cfg.problem.name}_dataset_{cfg.task}.pkl'
    dataset = {} if not dataset_path.exists() else pkl.load(open(dataset_path, 'rb'))
    time_dataset = []

    # For each size
    for size in cfg.size:
        if size in dataset:
            print("Overwriting dataset for size ", size, "... \n")
        dataset[size] = {}

        for split in cfg.split:
            dataset[size][split] = []
            df_label = pd.read_csv(res_path / 'labels' / cfg.problem.name / size / f'label_{size}_{split}.csv')
            print(f"Split {split}, labels shape {df_label.shape}")

            smac_out = res_path / 'smac_output' / cfg.problem.name / f'iinc_{cfg.iinc}' / size
            inst_path = inst_root_path / size / split
            for inst in inst_path.iterdir():
                pid = int(inst.stem.split('.')[0].split('_')[-1])
                sample = {'name': inst.stem, 'pid': pid, 'seed': None, 'x': {}, 'y': []}

                start_time = time.time()
                # Prepare x
                data = read_data_from_file(cfg.problem.acronym, inst)
                featurizer = featurizer_factory.create(cfg.featurizer.name, cfg=cfg.featurizer, data=data)
                features = featurizer.get()
                sample['x'] = features

                # Prepare y
                # Get best seed for the pid
                label_row = df_label[df_label['pid'] == pid]
                best_seed = label_row['seed'].values[0]
                sample['seed'] = best_seed
                # Load the traj json for that (pid, best_seed) combination
                smac_out_dir = smac_out / split / inst.stem / f'run_{best_seed}'
                traj = smac_out_dir.joinpath('traj.json')
                with open(traj, 'r') as fp:
                    lines = fp.readlines()
                    for line in lines[1:]:
                        run = json.loads(line)
                        sample['y'].append({'cost': np.asarray(run['cost']).reshape(1, 1),
                                            'pwt': property_weight_dict2array(run['incumbent'],
                                                                              cast_to_numpy=True)})
                    # If traj only contains one line
                    run = json.loads(lines[-1])
                    if len(sample['y']) == 0:
                        # Config didn't run for some reason
                        if run['cost'] > 10000:
                            sample['y'].append({'cost': np.asarray(np.infty).reshape(1, 1)})
                        # Config incurred PAR10 penalty
                        else:
                            sample['y'].append({'cost': np.asarray(run['cost'] / 10).reshape(1, 1)})
                        # Save property weights of the incumbent configuration
                        sample['y'][-1]['pwt'] = property_weight_dict2array(run['incumbent'], cast_to_numpy=True)

                    # Save rank of the incumbent configuration
                    sample['y'][-1]['rank'] = get_variable_rank_from_weights(data, run['incumbent'],
                                                                             normalized=bool(cfg.normalize_rank))
                end_time = time.time() - start_time
                time_dataset.append([size, pid, best_seed, split, end_time])

                # Append sample to dataset
                dataset[size][split].append(sample)

    # Save time
    time_df = pd.DataFrame(time_dataset, columns=["size", "pid", "best_seed", "split", "time"])
    time_df.to_csv(res_path / f'datasets/{cfg.problem.name}/{cfg.problem.name}_time_dataset.csv', index=False)

    # Save dataset
    with open(res_path / f'datasets/{cfg.problem.name}/{cfg.problem.name}_dataset.pkl', 'wb') as fp:
        pkl.dump(dataset, fp)


def generate_dataset_pair_svmrank(cfg):
    res_path = Path(cfg.res_path[cfg.machine])
    inst_root_path = res_path / 'instances' / cfg.problem.name
    time_dataset = []

    # For each size
    for size in cfg.size:
        for split in cfg.split:
            df_label = pd.read_csv(res_path / 'labels' / cfg.problem.name / size / f'label_{size}_{split}.csv')
            print(f"Split {split}, labels shape {df_label.shape}")

            split_str = ''
            n_items_str = ''
            inst_names_str = ''
            qid = 1
            inst_path = inst_root_path / size / split
            for inst in inst_path.iterdir():
                pid = int(inst.stem.split('.')[0].split('_')[-1])
                inst_names_str += f'{inst.stem}\n'

                start_time = time.time()

                # Get features
                data = read_data_from_file(cfg.problem.acronym, inst)
                featurizer = featurizer_factory.create(cfg.featurizer.name, cfg=cfg.featurizer, data=data)
                features = featurizer.get()

                # Get best seed for the pid
                label_row = df_label[df_label['pid'] == pid]
                best_seed = label_row['seed'].values[0]

                # Get variable order
                incb_dict = ast.literal_eval(label_row['incb'].values[0])
                # A lower ranks means the variable is used higher up in the DD construction
                # However, SVMRank needs rank to be higher for the variable to be used higher in DD construction
                # Hence, modified_rank = n_items - original_rank
                ranks = get_variable_rank_from_weights(data, incb_dict, normalized=bool(cfg.normalize_rank))
                n_items = len(ranks)
                n_items_str += f'{int(n_items)}\n'
                for item_id, r in enumerate(ranks):
                    fid = 1
                    features_str = ''

                    for f in features['var'][item_id]:
                        features_str += f"{fid}:{f} "
                        fid += 1

                    for f in features['vrank'][item_id]:
                        features_str += f"{fid}:{f} "
                        fid += 1

                    # Rank modified to be consistent with the convention of SVMRank
                    modified_rank = int(n_items - r)
                    split_str += f'{modified_rank} qid:{qid} {features_str}\n'

                # Update qid after processing one instance
                qid += 1

                end_time = time.time() - start_time
                time_dataset.append([size, pid, best_seed, split, end_time])

            # Save dataset
            fp = open(res_path / f'datasets/{cfg.problem.name}/{cfg.problem.name}_dataset_{cfg.task}_{split}.dat', 'w')
            fp.write(split_str)

            fp = open(res_path / f'datasets/{cfg.problem.name}/{cfg.problem.name}_n_items_{cfg.task}_{split}.dat', 'w')
            fp.write(n_items_str)

            fp = open(res_path / f'datasets/{cfg.problem.name}/{cfg.problem.name}_names_{cfg.task}_{split}.dat', 'w')
            fp.write(inst_names_str)

    # Save time
    time_df = pd.DataFrame(time_dataset, columns=["size", "pid", "best_seed", "split", "time"])
    time_df.to_csv(res_path / f'datasets/{cfg.problem.name}/{cfg.problem.name}_time_dataset_{cfg.task}.csv',
                   index=False)


@hydra.main(version_base='1.1', config_path='../config', config_name='generate_dataset.yaml')
def main(cfg: DictConfig):
    """Generate dataset

    Parameters
    ----------
    cfg

    Returns
    -------

    """
    if cfg.task == 'point_regress':
        generate_dataset_point_regress(cfg)
    elif cfg.task == 'multitask':
        generate_dataset_multitask(cfg)
    elif cfg.task == 'pair_svmrank':
        generate_dataset_pair_svmrank(cfg)


if __name__ == '__main__':
    main()
