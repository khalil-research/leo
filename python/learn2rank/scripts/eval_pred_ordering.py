import logging
import os
import pickle as pkl
from pathlib import Path

import hydra
from omegaconf import DictConfig

from learn2rank.utils.bdd import run_bdd_builder
from learn2rank.utils.order import make_result_column

# A logger for this file
log = logging.getLogger(__name__)

print(os.getcwd())


@hydra.main(version_base='1.1', config_path='../config', config_name='eval_pred_ordering.yaml')
def main(cfg: DictConfig):
    resource_path = Path(cfg.res_path[cfg.machine])
    inst_path = resource_path.joinpath(f'instances/{cfg.problem.name}')
    preds_path = resource_path.joinpath(f'predictions/{cfg.pred.name}')
    preds = pkl.load(open(str(preds_path), 'rb'))
    if cfg.split == 'train':
        preds = preds['tr']
    else:
        preds = preds['val']
    # preds = preds['val'] if cfg.split is None else preds[cfg.split]
    results = []

    names, n_items, order = preds['names'], preds['n_items'], preds['order']
    for _name, _n_item, _order in zip(names, n_items, order):
        _, _, n_objs, n_vars, pid = _name.split('_')
        pid = int(pid)
        if pid < cfg.from_pid or pid >= cfg.to_pid:
            continue

        log.info(f'Processing {_name}')

        size = f'{n_objs}_{n_vars}'
        dat_path = inst_path / size / f'{cfg.split}/{_name}.dat'

        status, result = run_bdd_builder(str(dat_path), _order[:_n_item], bin_path=str(resource_path),
                                         prob_id=str(cfg.problem.id), preprocess=str(cfg.problem.preprocess),
                                         time_limit=cfg.bdd.timelimit, mem_limit=cfg.bdd.memlimit)
        results.append(make_result_column(cfg.problem.name,
                                          size,
                                          cfg.split,
                                          pid,
                                          'pred',
                                          result))

    with open(f"eval_{cfg.pred.name.split('.')[0]}_{cfg.split}_{cfg.from_pid}_{cfg.to_pid}.pkl", 'wb') as fp:
        pkl.dump(results, fp)


if __name__ == '__main__':
    main()
