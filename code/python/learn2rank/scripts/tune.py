import logging
from pathlib import Path

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf

from learn2rank.model.factory import model_factory
from learn2rank.trainer.factory import trainer_factory
from learn2rank.utils import set_seed, set_machine
from learn2rank.utils.data import load_dataset

log = logging.getLogger(__name__)


class GradientBoostingRankerObj:
    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data

    def __call__(self, trial):
        # Set model params
        self.cfg.model.verbosity = 1
        self.cfg.model.n_estimators = trial.suggest_int("n_estimators", 75, 200, step=25)
        self.cfg.model.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1.0, log=True)
        self.cfg.model.gamma = trial.suggest_int("gamma", 0, 5)
        self.cfg.model.max_depth = trial.suggest_int("max_depth", 3, 9, step=2)
        self.cfg.model.min_child_weight = trial.suggest_int("min_child_weight", 0, 5)
        self.cfg.model.subsample = trial.suggest_float("subsample", 0.2, 1.0)
        self.cfg.model.colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 1.0)
        self.cfg.model.reg_lambda = trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True)
        self.cfg.model.grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        # if 'context' in self.cfg.task:
        #     self.cfg.model.feature_importance = trial.suggest_int("feature_importance", 0, 1)

        # Build model
        log.info(f'* Building model...')
        model = model_factory.create(self.cfg.model.name, cfg=self.cfg.model)
        log.info(OmegaConf.to_yaml(self.cfg.model, resolve=True))
        log.info('')

        # Build trainer and run
        log.info(f'* Starting trainer...')
        trainer = trainer_factory.create(self.cfg.model.trainer, model=model, data=self.data, cfg=self.cfg)
        trainer.run()

        return trainer.val_tau


@hydra.main(version_base='1.2', config_path='../config', config_name='tune.yaml')
def main(cfg: DictConfig):
    set_machine(cfg)

    log.info(f'* Learn2rank BDD: problem {cfg.problem.name}')
    log.info(f'* Script: tune.py')
    log.info(f'* Task: {cfg.task}, Fused: {cfg.dataset.fused}')
    if 'all' in cfg.task:
        cfg.dataset.fused = 1
    else:
        cfg.dataset.fused = 0
        log.info(f'* Size: {cfg.problem.size}')

    log.info(f'* Setting seed to {cfg.run.seed} for reproducibility \n')
    set_seed(cfg.run.seed)

    log.info(f'* Loading data...')
    log.info(f'* Dataset: {cfg.dataset.path}')
    data = load_dataset(cfg)

    study = optuna.create_study(direction='maximize')
    study.optimize(globals()[f'{cfg.model.name}Obj'](cfg, data),
                   n_trials=cfg.n_trials,
                   timeout=cfg.time_limit_study)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    cfg.val_tau = float(trial.val_tau)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        cfg.model[key] = value

    # Save the best model config with <model_id>.yaml
    model = model_factory.create(cfg.model.name, cfg=cfg.model)
    best_model_cfg_path = Path(cfg.res_path[cfg.machine]) / 'model_cfg' / f'{model.id}.yaml'
    best_model_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_model_cfg_path, "w") as fp:
        OmegaConf.save(cfg, fp)


if __name__ == '__main__':
    main()
