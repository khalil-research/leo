import pathlib

import hydra
from omegaconf import DictConfig, OmegaConf


def check_cfg_match(cfg, run_cfg):
    return True


@hydra.main(version_base=None, config_path='config', config_name='find_best_model.yaml')
def main(cfg: DictConfig):
    output_path = pathlib.Path(__file__).parent.joinpath('outputs')
    for f in output_path.rglob('*/config.yaml'):
        run_cfg = OmegaConf.load(f)
        is_match = check_cfg_match(cfg, run_cfg)
        if is_match:
            pass


if __name__ == '__main__':
    main()
