import ast
import pickle as pkl
import time

import hydra
import pandas as pd
from omegaconf import DictConfig

from leo import path
from leo.featurizer.factory import featurizer_factory
from leo.utils.data import get_dataset_name
from leo.utils.data import read_data_from_file
from leo.utils.order import get_variable_rank


def save_dataset_point_regress(root_path, dataset_name, task, split, dataset, time_dataset):
    with open(root_path / f'{dataset_name}_dataset_{task}_{split}.pkl', 'wb') as fp:
        pkl.dump(dataset, fp)

    time_df = pd.DataFrame(time_dataset, columns=['size', 'pid', 'best_seed', 'split', 'time'])
    time_df.to_csv(root_path / f'{dataset_name}_time_{task}_{split}.csv', index=False)


def save_dataset_pair_rank(root_path, dataset_name, task, split, split_str, n_items_str, inst_names_str, time_dataset):
    root_path.joinpath(f'{dataset_name}_dataset_{task}_{split}.dat').write_text(split_str)
    root_path.joinpath(f'{dataset_name}_n_items_{task}_{split}.dat').write_text(n_items_str)
    root_path.joinpath(f'{dataset_name}_names_{task}_{split}.dat').write_text(inst_names_str)

    time_df = pd.DataFrame(time_dataset, columns=['size', 'pid', 'best_seed', 'split', 'time'])
    time_df.to_csv(root_path / f'{dataset_name}_time_{task}_{split}.csv', index=False)


def generate_dataset_point_regress(cfg):
    inst_root_path = path.instances / cfg.problem.name
    featurizer = featurizer_factory.create(cfg.featurizer.name, cfg=cfg.featurizer)

    # For each size
    for split in cfg.split:
        print(f'Split: {split}')
        print(f'Fused: {cfg.fused}')
        dataset_name = get_dataset_name(cfg) if cfg.fused else None
        dataset, time_dataset = [], []

        for size in cfg.size:
            print('\tSize: ', size)
            n_objs, n_vars = list(map(int, size.split('_')))
            cfg.problem.n_objs = n_objs
            cfg.problem.n_vars = n_vars
            if not cfg.fused:
                dataset_name = get_dataset_name(cfg)

            df_label = None
            if split != 'test':
                df_label = pd.read_csv(path.label / cfg.problem.name / size / f'label_{size}_{split}.csv')
                print(f'\t\tLabels shape: {df_label.shape}')

            inst_path = inst_root_path / size / split
            for inst in inst_path.iterdir():
                pid = int(inst.stem.split('.')[0].split('_')[-1])
                sample = {'name': inst.stem, 'pid': pid, 'seed': None, 'x': {}, 'y': []}

                # Prepare x
                data = read_data_from_file(cfg.problem.acronym, inst)
                start_time = time.time()
                features = featurizer.get(data=data)
                sample['x'] = features

                # Prepare y
                # Get best seed for the pid
                best_seed = None
                if df_label is not None:
                    label_row = df_label[df_label['pid'] == pid]
                    # Use min_weight if label not found
                    incb_dict = {'avg_value': 0.0, 'avg_value_by_weight': 0.0, 'max_value': 0.0,
                                 'max_value_by_weight': 0.0,
                                 'min_value': 0.0, 'min_value_by_weight': 0.0, 'weight': -1.0}
                    best_seed = '-1'
                    sample['seed'] = '-1'
                    if label_row.shape[0]:
                        best_seed = label_row['seed'].values[0]
                        sample['seed'] = best_seed
                        # Save the incumbent
                        incb_dict = ast.literal_eval(label_row['incb'].values[0])
                    else:
                        print(f'Missing incumbent. Using min_weight for {str(inst)}')
                    # Top variables get a lower rank. For example, first variable is ranked 0
                    sample['y'] = get_variable_rank(data=data, property_weights=incb_dict, reverse=True,
                                                    normalized=bool(cfg.normalize_rank))[0]

                end_time = time.time() - start_time
                time_dataset.append([size, pid, best_seed, split, end_time])

                # Append sample to dataset
                dataset.append(sample)

            if not cfg.fused:
                dataset_root_path = path.dataset / cfg.problem.name
                dataset_root_path.mkdir(exist_ok=True, parents=True)
                save_dataset_point_regress(dataset_root_path, dataset_name, cfg.task, split, dataset, time_dataset)
                dataset, time_dataset = [], []

        if cfg.fused:
            dataset_root_path = path.dataset / cfg.problem.name
            dataset_root_path.mkdir(exist_ok=True, parents=True)
            save_dataset_point_regress(dataset_root_path, dataset_name, cfg.task, split, dataset, time_dataset)


def generate_dataset_pair_rank(cfg):
    inst_root_path = path.instances / cfg.problem.name
    featurizer = featurizer_factory.create(cfg.featurizer.name, cfg=cfg.featurizer)

    dataset_root_path = path.dataset / cfg.problem.name
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    # For each split
    for split in cfg.split:
        print(f'Split: {split}')
        print(f'Fused: {cfg.fused}')
        dataset_name = get_dataset_name(cfg) if cfg.fused else None
        split_str, n_items_str, inst_names_str, qid = '', '', '', 1
        time_dataset = []

        # For each size
        for size in cfg.size:
            print('\tSize: ', size)
            n_objs, n_vars = list(map(int, size.split('_')))
            cfg.problem.n_objs = n_objs
            cfg.problem.n_vars = n_vars
            if not cfg.fused:
                dataset_name = get_dataset_name(cfg)

            df_label = None
            if split != 'test':
                df_label = pd.read_csv(path.label / cfg.problem.name / size / f'label_{size}_{split}.csv')
                print(f'\t\tLabels shape: {df_label.shape}')

            inst_path = inst_root_path / size / split
            for inst in inst_path.iterdir():
                pid = int(inst.stem.split('.')[0].split('_')[-1])
                inst_names_str += f'{inst.stem}\n'
                n_items_str += f'{int(n_vars)}\n'

                # Get features
                data = read_data_from_file(cfg.problem.acronym, inst)

                start_time = time.time()
                features = featurizer.get(data=data)

                # Get best seed for the pid
                ranks, best_seed = None, None
                if df_label is not None:
                    label_row = df_label[df_label['pid'] == pid]
                    best_seed = label_row['seed'].values[0]

                    # Get variable order
                    incb_dict = ast.literal_eval(label_row['incb'].values[0])
                    # A lower ranks means the variable is used higher up in the DD construction
                    # However, SVMRank needs rank to be higher for the variable to be used higher in DD construction
                    # Hence, modified_rank = n_items - original_rank
                    ranks = get_variable_rank(data=data, property_weights=incb_dict, reverse=True,
                                              normalized=bool(cfg.normalize_rank))[0]
                # For the test set
                ranks = [n_vars] * n_vars if ranks is None else ranks
                for item_id, r in enumerate(ranks):
                    fid = 1
                    features_str = ''

                    for f in features['var'][item_id]:
                        features_str += f"{fid}:{f} "
                        fid += 1

                    for f in features['vrank'][item_id]:
                        features_str += f"{fid}:{f} "
                        fid += 1

                    if cfg.context:
                        for f in features['inst'][item_id]:
                            features_str += f"{fid}:{f} "
                            fid += 1

                    # Rank modified to be consistent with the convention of SVMRank
                    modified_rank = int(n_vars - r)
                    split_str += f'{modified_rank} qid:{qid} {features_str}\n'

                # Update qid after processing one instance
                qid += 1

                end_time = time.time() - start_time
                time_dataset.append([size, pid, best_seed, split, end_time])

            # Separate dataset for each size
            if not cfg.fused:
                save_dataset_pair_rank(dataset_root_path, dataset_name, cfg.task, split, split_str, n_items_str,
                                       inst_names_str, time_dataset)
                # Reset
                split_str, n_items_str, inst_names_str, qid = '', '', '', 1
                time_dataset = []

        # Single dataset for all sizes
        if cfg.fused:
            save_dataset_pair_rank(dataset_root_path, dataset_name, cfg.task, split, split_str, n_items_str,
                                   inst_names_str, time_dataset)


@hydra.main(version_base='1.2', config_path='./config', config_name='generate_dataset.yaml')
def main(cfg: DictConfig):
    """Generate dataset

    Parameters
    ----------
    cfg

    Returns
    -------

    """
    print(f'Task: {cfg.task}, Fused: {cfg.fused}, Context: {cfg.context}')
    cfg.featurizer.context = int(bool(cfg.context))

    if cfg.task == 'point_regress':
        generate_dataset_point_regress(cfg)

    elif cfg.task == 'pair_rank':
        generate_dataset_pair_rank(cfg)

    else:

        raise ValueError(f'Invalid task name: {cfg.task}')


if __name__ == '__main__':
    main()
