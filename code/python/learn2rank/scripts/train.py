import logging
import pickle as pkl
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from learn2rank.model.factory import model_factory
from learn2rank.trainer.factory import trainer_factory
from learn2rank.utils import set_seed

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base='1.1', config_path='../config', config_name='train.yaml')
def main(cfg: DictConfig):
    log.info(f'* Learn2rank BDD: problem {cfg.problem.name}, '
             f'max objectives {cfg.problem.n_max_objs}, '
             f'max variables {cfg.problem.n_max_vars}\n')

    log.info(f'* Setting seed to {cfg.run.seed} for reproducibility \n')
    set_seed(cfg.run.seed)

    log.info(f'* Building model...')
    model = model_factory.create(cfg.model.name, cfg=cfg.model)
    log.info(OmegaConf.to_yaml(cfg.model, resolve=True))
    log.info('')

    log.info(f'* Loading data...')
    # Dataset path

    # Load data if pickled. Otherwise, pass the path as data
    dp = Path(cfg.dataset.path)
    if str(dp).split('.')[-1] == 'pkl':
        fp = open(dp, 'rb')
        data = pkl.load(fp)
    else:
        data = str(dp)
    log.info(dp)
    log.info('')

    log.info(f'* Starting trainer...')
    trainer = trainer_factory.create(cfg.model.trainer, model=model, data=data, cfg=cfg)
    trainer.run()


if __name__ == '__main__':
    main()
