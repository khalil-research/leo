import logging
import pickle as pkl
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from learn2rank.utils.data import load_svmlight_data
from learn2rank.model.factory import model_factory
from learn2rank.trainer.factory import trainer_factory
from learn2rank.utils import set_seed

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


def update_model_cfg(cfg, params):
    # TODO: Add params of other models
    if type(params) == str and len(params.strip()):
        for p in params.split('_'):
            k, v = p.split('-')
            v = v.strip()
            if k == 'nes':
                cfg.model.n_estimators = int(v)
            elif k == 'md':
                cfg.model.max_depth = int(v)
            elif k == 'lr':
                cfg.model.learning_rate = float(v)
            elif k == 'gma':
                cfg.model.gamma = int(v)
            elif k == 'mcw':
                cfg.model.min_child_weight = int(v)
            elif k == 'rlam':
                cfg.model.reg_lambda = float(v)
            elif k == 'raph':
                cfg.model.reg_alpha = float(v)
            elif k == 'ss':
                cfg.model.subsample = float(v)
            elif k == 'csbt':
                cfg.model.colsample_bytree = float(v)
            elif k == 'gp':
                cfg.model.grow_policy = str(v)

    return cfg


# def load_svmlight_data(files, split_types, file_types):
#     i = 0
#     data = {}
#     for st in split_types:
#         for ft in file_types:
#             if ft == 'dataset':
#                 data[st, ft] = load_svmlight_file(str(files[i])) if files[i].exists() else None
#             elif ft == 'n_items':
#                 data[st, ft] = list(map(int, files[i].read_text().strip().split('\n'))) \
#                     if files[i].exists() else None
#             elif ft == 'names':
#                 data[st, ft] = files[i].read_text().strip().split('\n') if files[i].exists() else None
#
#             i += 1
#
#     return data


def load_model(cfg, model_path_prefix, best_model_row):
    prefix = model_prefix.get(cfg.model.name)
    params = best_model_row.iloc[0]['model_params']
    cfg = update_model_cfg(cfg, params)
    ext = model_ext.get(cfg.model.name)

    model = model_factory.create(cfg.model.name, cfg=cfg.model)
    if ext is not None:
        model_id = f"{prefix}_{params}"
        model_path = model_path_prefix / f'model_{model_id}{ext}'

        if ext == '.pkl':
            model = pkl.load(open(model_path, 'rb'))
        elif ext == '.txt' and 'Ranker' in cfg.model.name:
            model.load_model(model_path)

    return model


@hydra.main(version_base='1.2', config_path='../config', config_name='test.yaml')
def main(cfg: DictConfig):
    log.info(f'* Setting seed to {cfg.run.seed} for reproducibility')
    set_seed(cfg.run.seed)

    log.info(f'* Task: {cfg.task}, Fused: {cfg.dataset.fused}')
    if not cfg.dataset.fused:
        log.info(f'* Size: {cfg.problem.size}')

    # Set paths
    resource_path = Path(cfg.res_path[cfg.machine])
    path_suffix = Path(cfg.problem.name)
    if cfg.dataset.fused and 'context' not in cfg.task:
        path_suffix /= 'all'
    elif cfg.dataset.fused and 'context' in cfg.task:
        path_suffix /= 'all_context'
    else:
        path_suffix /= cfg.problem.size

    model_path_prefix = resource_path / 'pretrained' / path_suffix
    pred_path_prefix = resource_path / 'predictions' / path_suffix
    path_suffix_best = path_suffix.parent / f'best_model_{path_suffix.stem}.csv'
    best_model_summary_path = resource_path / 'model_summary' / path_suffix_best

    # Read best model summary
    df_best_model = pd.read_csv(best_model_summary_path, index_col=False)
    assert cfg.model.name in df_best_model['model_name'].values
    best_model_row = df_best_model[df_best_model['model_name'] == cfg.model.name]

    # Load model
    model = load_model(cfg, model_path_prefix, best_model_row)

    # Load data
    dp = Path(cfg.dataset.path)
    data = None
    data = pkl.load(open(dp, 'rb')) if dp.suffix == 'pkl' else data
    if data is None and 'rank' in cfg.task:
        split_types = ['train', 'val', 'test']
        file_types = ['dataset', 'n_items', 'names']

        suffix = 'pair_svmrank'
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

    prediction_path = pred_path_prefix / best_model_row.iloc[0]['prediction_path']
    pred_store = pkl.load(open(prediction_path, 'rb'))

    result_path = pred_path_prefix / best_model_row.iloc[0]['results_path']
    result_store = pkl.load(open(result_path, 'rb'))

    log.info(f'* Starting trainer...')
    trainer = trainer_factory.create(cfg.model.trainer, model=model, data=data, cfg=cfg,
                                     ps=pred_store, rs=result_store)
    trainer.predict(split='test')


if __name__ == '__main__':
    main()
