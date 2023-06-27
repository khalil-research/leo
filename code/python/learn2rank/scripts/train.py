import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from learn2rank.model.factory import model_factory
from learn2rank.trainer.factory import trainer_factory
from learn2rank.utils import set_seed, set_machine
from learn2rank.utils.data import load_dataset

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base='1.2', config_path='../config', config_name='train.yaml')
def main(cfg: DictConfig):
    set_machine(cfg)

    log.info(f'* Learn2rank BDD: problem {cfg.problem.name}')
    log.info(f'* Script: train.py')
    log.info(f'* Task: {cfg.task}, Fused: {cfg.dataset.fused}')
    if not cfg.dataset.fused:
        log.info(f'* Size: {cfg.problem.size}')

    log.info(f'* Setting seed to {cfg.run.seed} for reproducibility \n')
    set_seed(cfg.run.seed)

    log.info(f'* Loading data...')
    log.info(f'* Dataset: {cfg.dataset.path}')
    data = load_dataset(cfg)

    log.info(f'* Building model...')
    if cfg.tuned_model is not None:
        pass
    model = model_factory.create(cfg.model.name, cfg=cfg.model)
    log.info(OmegaConf.to_yaml(cfg.model, resolve=True))
    log.info('')

    log.info(f'* Starting trainer...')
    trainer = trainer_factory.create(cfg.model.trainer, model=model, data=data, cfg=cfg)
    trainer.run()
    cfg.val_tau = float(trainer.val_tau)

    model_cfg_path = Path(cfg.res_path[cfg.machine]) / 'model_cfg' / f'{model.id}.yaml'
    if not model_cfg_path.exists():
        model_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_cfg_path, "w") as fp:
            OmegaConf.save(cfg.model, fp)

    print('val_tau: ', trainer.val_tau)


if __name__ == '__main__':
    main()
