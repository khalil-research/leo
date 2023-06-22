import logging
import pickle as pkl
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from learn2rank.model.factory import model_factory
from learn2rank.trainer.factory import trainer_factory
from learn2rank.utils import set_seed, set_machine
from learn2rank.utils.data import load_svmlight_data

# A logger for this file
log = logging.getLogger(__name__)

model_prefix = {
    'NeuralRankingMachine': 'nrm',
    'LinearRegression': 'lr',
    'Ridge': 'ridge',
    'Lasso': 'lasso',
    'DecisionTreeRegressor': 'dtree',
    'GradientBoostingRegressor': 'gbr',
    'GradientBoostingRanker': 'xgb',
    'SVMRank': 'svmrank',
    'MinWeight': 'minwt',
    'SmacOne': 'smac',
    'SmacAll': 'smac_all',
    'Canonical': 'canonical',
}

model_ext = {
    'NeuralRankingMachine': '.pkl',
    'LinearRegression': '.pkl',
    'Ridge': '.pkl',
    'Lasso': '.pkl',
    'DecisionTreeRegressor': '.pkl',
    'GradientBoostingRegressor': '.pkl',
    'GradientBoostingRanker': '.txt',
    'SVMRank': '.dat',
    'MinWeight': None,
    'SmacOne': None,
    'SmacAll': None,
    'Canonical': None,
}


def load_model(cfg, model_path_prefix, model_id, model_name):
    ext = model_ext.get(model_name)
    model_path = model_path_prefix / f"model_{model_id}{ext}"
    model = model_factory.create(cfg.model.name, cfg=cfg.model)
    if ext is not None:
        if ext == '.pkl':
            model = pkl.load(open(model_path, 'rb'))
        elif ext == '.txt' and 'Ranker' in cfg.model.name:
            model.load_model(model_path)

    return model


@hydra.main(version_base='1.2', config_path='../config', config_name='test.yaml')
def main(cfg: DictConfig):
    set_machine(cfg)

    log.info(f'* Setting seed to {cfg.run.seed} for reproducibility')
    set_seed(cfg.run.seed)

    if '_all' in cfg.task:
        cfg.dataset.fused = 1
        size = 'all'
    else:
        cfg.dataset.fused = 0
        size = cfg.problem.size
    log.info(f'* Task: {cfg.task}, Size: {size}, Fused: {cfg.dataset.fused}')

    # Set paths
    resource_path = Path(cfg.res_path[cfg.machine])
    path_suffix = Path(cfg.problem.name)
    if cfg.dataset.fused and 'context' not in cfg.task:
        path_suffix /= 'all'
    elif cfg.dataset.fused and 'context' in cfg.task:
        path_suffix /= 'all_context'
    else:
        path_suffix /= cfg.problem.size

    # Set model_ids, model_names and tasks
    # models_ids, model_names, tasks = [], [], []
    model_ids, model_names = [], []
    if cfg.mode == 'all':
        # path_suffix = path_suffix.parent / f'{path_suffix}.csv'
        summary_path = resource_path / 'model_summary' / f'{path_suffix}.csv'
        df_summary = pd.read_csv(summary_path, index_col=False)
        df_summary = df_summary if cfg.task is None else df_summary.query(f"task == '{cfg.task}'")

        model_ids = df_summary.model_id.values
        model_names = df_summary.model_name.values
        # tasks = df_summary.task.values

    elif cfg.mode == 'one':
        # path_suffix = path_suffix.parent / f'summary.csv'
        summary_path = resource_path / 'model_summary' / 'summary.csv'
        df_summary = pd.read_csv(summary_path, index_col=False)
        df_summary = df_summary.query(f"model_id == '{cfg.model_id}'")
        cfg.task = df_summary.iloc[0]['task']

        model_ids = [cfg.model_id]
        model_names = [df_summary.iloc[0]['model_name']]
        # tasks = [cfg.task]

    elif cfg.mode == 'best':
        # path_suffix = path_suffix.parent / f'best_model_{path_suffix.stem}.csv'
        summary_path = resource_path / 'model_summary' / f'best_model_{path_suffix.stem}.csv'
        df_summary = pd.read_csv(summary_path, index_col=False)
        df_summary = df_summary if cfg.task is None else df_summary.query(f"task == '{cfg.task}'")

        model_ids = df_summary.model_id.values
        model_names = df_summary.model_name.values
        # tasks = df_summary.task.values

    counter = 1
    model_cfg_path_prefix = resource_path / 'model_cfg'
    # for task in set(tasks):

    # Load data
    dp = Path(cfg.dataset.path)
    data = pkl.load(open(dp, 'rb')) if dp.suffix == '.pkl' else None
    if data is None and 'rank' in cfg.task:
        split_types = ['train', 'val', 'test']
        file_types = ['dataset', 'n_items', 'names']

        suffix = 'pair_rank'
        if cfg.dataset.fused and 'context' not in cfg.task:  # Load fused dataset
            suffix += '_all'
        elif cfg.dataset.fused and 'context' in cfg.task:  # Load fused dataset
            suffix += '_all_context'

        splits_suffix = list(map(lambda x: f'{suffix}_{x}.dat', split_types))
        files_prefix = [f'{ft}_{ss}' for ss in splits_suffix for ft in file_types]
        if not cfg.dataset.fused:
            files_prefix = [f'{cfg.problem.size}_{fp}' for fp in files_prefix]
        files = [dp / fp for fp in files_prefix]
        data = load_svmlight_data(files, split_types, file_types)

    # Select models to run
    # _selected = [(x, y) for x, y, z in zip(model_ids, model_names, tasks) if z == task]
    for model_id, model_name in zip(model_ids, model_names):
        print(counter, model_name, model_id)

        counter += 1

        model_path_prefix = resource_path / 'pretrained' / cfg.problem.name / path_suffix.stem
        model_cfg = OmegaConf.load(model_cfg_path_prefix / f"{model_id}.yaml")

        # Load model
        model = load_model(model_cfg, model_path_prefix, model_id, model_name)

        # Load previous predictions and result
        pred_path_prefix = resource_path / 'predictions' / cfg.problem.name / path_suffix.stem
        prediction_path = pred_path_prefix / f"prediction_{model_id}.pkl"
        pred_store = pkl.load(open(prediction_path, 'rb'))
        result_path = pred_path_prefix / f"results_{model_id}.pkl"
        result_store = pkl.load(open(result_path, 'rb'))

        # Predict on test set
        log.info(f'* Starting trainer...')
        trainer = trainer_factory.create(model_cfg.model.trainer, model=model, data=data, cfg=model_cfg,
                                         ps=pred_store, rs=result_store)
        trainer.predict(split='test')


if __name__ == '__main__':
    main()
