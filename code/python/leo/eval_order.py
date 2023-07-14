import logging
import pickle as pkl

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from leo import path
from leo.utils.bdd import run_bdd_builder
from leo.utils.data import get_dataset_name
from leo.utils.order import make_result_column

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base='1.2', config_path='./config', config_name='eval_order.yaml')
def main(cfg: DictConfig):
    dataset_name = get_dataset_name(cfg)
    pred_path = path.prediction / cfg.problem.name / dataset_name

    if cfg.mode == 'one':
        assert cfg.model_id is not None

        model_summary_path = path.model_summary / cfg.problem.name / f'{dataset_name}.csv'
        df_summary = pd.read_csv(model_summary_path, index_col=False)
        row = df_summary.query("model_id == '{}'".format(cfg.model_id))
    elif cfg.mode == 'best':
        assert cfg.model_name is not None

        model_summary_path = path.model_summary / cfg.problem.name / f'best_model_{dataset_name}.csv'
        df_summary = pd.read_csv(model_summary_path, index_col=False)
        row = df_summary[(df_summary.model_name == cfg.model_name) & (df_summary.task == cfg.task)]
    else:
        raise ValueError('Invalid mode!')

    results = []
    prediction_path = pred_path / f"prediction_{row.iloc[0]['model_id']}.pkl"
    preds = pkl.load(open(str(prediction_path), 'rb'))
    names, n_items, order = preds[cfg.split]['names'], preds[cfg.split]['n_items'], preds[cfg.split]['order']
    for _name, _n_item, _order in zip(names, n_items, order):
        _, _, n_objs, n_vars, pid = _name.split('_')
        size = f'{n_objs}_{n_vars}'
        pid = int(pid)
        if pid < cfg.from_pid or pid >= cfg.to_pid:
            continue

        log.info(f'Processing {_name}')
        dat_path = path.instances / cfg.problem.name / size / f'{cfg.split}/{_name}.dat'
        status, result = run_bdd_builder(str(dat_path), _order[:_n_item], bin_path=str(path.bin),
                                         prob_id=str(cfg.problem.id), preprocess=str(cfg.problem.preprocess),
                                         time_limit=cfg.bdd.timelimit, mem_limit=cfg.bdd.memlimit)
        log.info(f'Time : {np.sum(result[-4: -1])}')
        results.append(make_result_column(cfg.problem.name,
                                          size,
                                          cfg.split,
                                          pid,
                                          cfg.task,
                                          f'pred_{row.model_name.values[0]}',
                                          result))

    df = pd.DataFrame(results, columns=['problem', 'size', 'split', 'pid', 'task', 'order_type', 'run_id',
                                        'nnds', 'iw', 'rw', 'inc', 'rnc', 'iac', 'rac', 'iid', 'rid',
                                        'comp', 'comp_time', 'red_time', 'pareto_time', 'nnds_per_layer'])

    eval_order_path = path.eval_order / dataset_name
    eval_order_path.mkdir(parents=True, exist_ok=True)
    eval_order_path = eval_order_path / f'pred_{cfg.split}_{cfg.from_pid}_{cfg.to_pid}.csv'
    if eval_order_path.exists():
        df_old = pd.read_csv(eval_order_path)
        df = pd.concat([df, df_old], ignore_index=True)
    df.to_csv(eval_order_path, index=False)


if __name__ == '__main__':
    main()
