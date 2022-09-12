import logging
import pickle as pkl
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from learn2rank.trainer.factory import trainer_factory

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base='1.1', config_path='../config', config_name='test.yaml')
def main(cfg: DictConfig):
    log.info('* Testing pretrained models...')
    log.info(f'* Learn2rank BDD: problem {cfg.problem.name}, '
             f'max objectives {cfg.problem.n_max_objs}, '
             f'max variables {cfg.problem.n_max_vars}\n')

    log.info(f'* Loading model...')
    model_path = Path(f"./learn2rank/resources/pretrained/'{cfg.pretrained_dir}'/final.ckpt")
    final_ckpt = torch.load(model_path, map_location=cfg.map_location)

    # Load best model
    best_model = final_ckpt['model']
    # model = model_factory.create(cfg.model.name, cfg=cfg.model)
    # log.info(OmegaConf.to_yaml(cfg.model, resolve=True))
    # log.info('')

    log.info(f'* Loading data...')
    # Dataset path
    dp = Path(get_original_cwd()) / 'learn2rank/resources/datasets' / f"{cfg.problem.name}.pkl"
    with open(dp, 'rb') as fp:
        data = pkl.load(fp)
    log.info(dp)
    log.info('')

    log.info(f'* Starting trainer...')
    trainer = trainer_factory.create(cfg.model.trainer, model=best_model, data=data, cfg=cfg)
    trainer.predict()


if __name__ == '__main__':
    main()
