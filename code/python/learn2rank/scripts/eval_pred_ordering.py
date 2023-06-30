import logging
import os
import pickle as pkl
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from learn2rank.utils import set_machine
from learn2rank.utils.bdd import run_bdd_builder
from learn2rank.utils.order import make_result_column

# A logger for this file
log = logging.getLogger(__name__)

print(os.getcwd())


@hydra.main(version_base='1.2', config_path='../config', config_name='eval_pred_ordering.yaml')
def main(cfg: DictConfig):
    set_machine(cfg)
    resource_path = Path(cfg.res_path[cfg.machine])
    inst_path = Path(cfg.inst_path)

    suffix = cfg.problem.size
    if cfg.task == 'pair_rank_all':
        suffix = 'all'
    elif cfg.task == 'pair_rank_all_context':
        suffix = 'all_context'
    pred_path = Path(cfg.pred_path) / suffix

    if cfg.mode == 'all':
        assert cfg.model_id is not None

        model_summary_path = Path(cfg.model_summary_path) / f'summary.csv'
        df_summary = pd.read_csv(model_summary_path, index_col=False)
        row = df_summary.query("model_id == '{}'".format(cfg.model_id))
    elif cfg.mode == 'best':
        assert cfg.model_name is not None

        model_summary_path = Path(cfg.model_summary_path) / f'best_model_{suffix}.csv'
        df_summary = pd.read_csv(model_summary_path, index_col=False)
        row = df_summary.query("model_name == '{}'".format(cfg.model_name))
    else:
        raise ValueError('Invalid mode!')

    results = []
    # for model_name in model_names:
    # row = df_best_model[df_best_model['model_name'] == model_name]
    prediction_path = pred_path / f"prediction_{row.iloc[0]['model_id']}.pkl"
    # predictions = pkl.load(open(prediction_path, 'rb'))

    preds = pkl.load(open(str(prediction_path), 'rb'))
    names, n_items, order = preds[cfg.split]['names'], preds[cfg.split]['n_items'], preds[cfg.split]['order']
    for _name, _n_item, _order in zip(names, n_items, order):
        _, _, n_objs, n_vars, pid = _name.split('_')
        pid = int(pid)
        if pid < cfg.from_pid or pid >= cfg.to_pid:
            continue

        log.info(f'Processing {_name}')
        dat_path = inst_path / cfg.problem.size / f'{cfg.split}/{_name}.dat'

        status, result = run_bdd_builder(str(dat_path), _order[:_n_item], bin_path=str(resource_path),
                                         prob_id=str(cfg.problem.id), preprocess=str(cfg.problem.preprocess),
                                         time_limit=cfg.bdd.timelimit, mem_limit=cfg.bdd.memlimit)
        results.append(make_result_column(cfg.problem.name,
                                          cfg.problem.size,
                                          cfg.split,
                                          pid,
                                          f'pred_{row.model_name.values[0]}',
                                          result))

    eval_order_path = Path(cfg.eval_order_path) / suffix
    eval_order_path.mkdir(parents=True, exist_ok=True)
    eval_order_path = eval_order_path / f"pred_{cfg.split}_{cfg.from_pid}_{cfg.to_pid}.csv"

    df = pd.DataFrame(results, columns=['problem', 'size', 'split', 'pid', 'order_type', 'run_id',
                                        'nnds', 'iw', 'rw', 'inc', 'rnc', 'iac', 'rac', 'iid', 'rid',
                                        'comp', 'comp_time', 'red_time', 'pareto_time', 'nnds_per_layer'])
    if eval_order_path.exists():
        df_old = pd.read_csv(eval_order_path)
        df = pd.concat([df, df_old], ignore_index=True)

    df.to_csv(eval_order_path, index=False)


if __name__ == '__main__':
    main()
