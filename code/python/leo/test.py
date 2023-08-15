import logging
import pickle as pkl

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from leo import path
from leo.model.factory import model_factory
from leo.trainer.factory import trainer_factory
from leo.utils import set_seed
from leo.utils.data import get_dataset_name
from leo.utils.data import load_dataset

log = logging.getLogger(__name__)

model_ext = {
    'NeuralRankingMachine': '.pkl',
    'LinearRegression': '.pkl',
    'Ridge': '.pkl',
    'Lasso': '.pkl',
    'DecisionTreeRegressor': '.pkl',
    'GradientBoostingRegressor': '.pkl',
    'GradientBoostingRanker': '.txt',
    'SVMRank': '.dat',
    'HeuristicWeight': None,
    'HeuristicValue': None,
    'HeuristicValueByWeight': None,
    'SmacI': None,
    'SmacD': None,
    'Lex': None
}


def load_model(model_cfg, model_path_prefix, model_id, model_name):
    ext = model_ext.get(model_name)
    model_path = model_path_prefix / f"model_{model_id}{ext}"
    model = model_factory.create(model_cfg.name, cfg=model_cfg)

    if ext is not None:
        if ext == '.pkl':
            model = pkl.load(open(model_path, 'rb'))
        elif ext == '.txt' and 'Ranker' in model_cfg.name:
            model.load_model(model_path)

    return model


@hydra.main(version_base='1.2', config_path='./config', config_name='test.yaml')
def main(cfg: DictConfig):
    log.info(f'* Task: {cfg.task}, Fused: {cfg.fused}, Context: {cfg.context}')
    log.info(f'* Setting seed to {cfg.run.seed} for reproducibility')
    set_seed(cfg.run.seed)

    dataset_name = get_dataset_name(cfg)
    df_name = f'best_{dataset_name}' if cfg.mode == 'best' else dataset_name
    df_name += '.csv'

    df_summary = pd.read_csv(path.model_summary / cfg.problem.name / df_name)
    if cfg.mode == 'one':
        assert cfg.model_id is not None and cfg.task is not None, 'Please provide model id and task'
        model_ids = [cfg.model_id]
        model_names = df_summary[df_summary['model_id'] == cfg.model_id].model_names.values
        tasks = [cfg.task]
    else:
        model_ids, model_names, tasks = df_summary.model_id.values, df_summary.model_name.values, df_summary.task.values

    # Load data
    cached_dataset = {'pair_rank': None, 'point_regress': None}
    counter = 1
    for model_id, model_name, task in zip(model_ids, model_names, tasks):
        if cfg.task is not None and cfg.task == task:
            print(counter, model_name, model_id)
            counter += 1

            # Load model conf to conf and load model
            model_path_prefix = path.pretrained / cfg.problem.name / dataset_name
            model_cfg = OmegaConf.load(path.model_cfg / f"{model_id}.yaml")
            cfg_dict = OmegaConf.to_container(cfg)
            model_cfg_dict = OmegaConf.to_container(model_cfg)
            cfg_dict['model'] = model_cfg_dict
            cfg = OmegaConf.create(cfg_dict)
            cfg.task = task
            # Load model
            model = load_model(model_cfg, model_path_prefix, model_id, model_name)

            # Load dataset
            if cached_dataset[cfg.task] is None:
                data = load_dataset(cfg)
                cached_dataset[cfg.task] = data
            else:
                data = cached_dataset[cfg.task]

            # Load previous predictions and result
            pred_path_prefix = path.prediction / cfg.problem.name / dataset_name
            prediction_path = pred_path_prefix / f"prediction_{model_id}.pkl"
            pred_store = pkl.load(open(prediction_path, 'rb'))
            result_path = pred_path_prefix / f"results_{model_id}.pkl"
            result_store = pkl.load(open(result_path, 'rb'))

            # Predict on test set
            log.info(f'* Starting trainer...')
            trainer = trainer_factory.create(cfg.model.trainer, model=model, data=data, cfg=cfg,
                                             ps=pred_store, rs=result_store)
            trainer.predict(split='test')


if __name__ == '__main__':
    main()
