import pickle as pkl

import hydra
from omegaconf import DictConfig, OmegaConf

from learn2rank.model.factory import model_factory
from learn2rank.trainer.factory import trainer_factory
from learn2rank.utils import set_seed


@hydra.main(version_base='1.1', config_path='../config', config_name='train.yaml')
def main(cfg: DictConfig):
    print(f'* Learn2rank BDD: problem {cfg.problem.name}, '
          f'max objectives {cfg.problem.n_max_objs}, '
          f'max variables {cfg.problem.n_max_vars}\n')

    print(f'* Setting seed to {cfg.run.seed} for reproducibility \n')
    set_seed(cfg.run.seed)

    print(f'* Building model...')
    model = model_factory.create(cfg.model.name, cfg=cfg.model)
    print(OmegaConf.to_yaml(cfg.model, resolve=True))
    print()

    print(f'* Loading data...')
    with open(cfg.paths.dataset, 'rb') as fp:
        data = pkl.load(fp)
    print(cfg.paths.dataset)
    print()

    print(f'* Starting trainer...')
    trainer = trainer_factory.create(cfg.model.trainer, model=model, data=data, cfg=cfg)
    trainer.run()


# def test(cfg: DictConfig):
#     # Load model
#     model = load_model(cfg.model.path)
#     data = get_dummy_data(cfg.problem)
#     trainer = trainer_factory(cfg.model, model=model, data=data, cfg=cfg)
#
#     preds = trainer.predict()
#
#     # save preds


if __name__ == '__main__':
    main()
