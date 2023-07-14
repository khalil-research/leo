import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from leo.model.factory import model_factory
from leo.trainer.factory import trainer_factory
from leo.utils import set_seed
from leo.utils.data import load_dataset
from leo.utils.data import save_model_config

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base='1.2', config_path='./config', config_name='train.yaml')
def main(cfg: DictConfig):
    print()
    log.info(f'* Start LEO training,  problem {cfg.problem.name}')
    log.info(f'* Task: {cfg.task}, Fused: {cfg.fused}, Context: {cfg.context}')
    log.info(f'* Model: {cfg.model.name}')
    if not cfg.fused:
        log.info(f'* Size: {cfg.problem.size}')

    log.info(f'* Setting seed to {cfg.run.seed} for reproducibility \n')
    set_seed(cfg.run.seed)

    log.info(f'* Loading data...')
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
    save_model_config(cfg, model.id)
    print('val_tau: ', trainer.val_tau)
    print('**************************** FINISHED TRAINING ****************************\n\n')


if __name__ == '__main__':
    main()
