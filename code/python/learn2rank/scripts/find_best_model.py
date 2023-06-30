import pickle as pkl
from pathlib import Path

import hydra
import numpy as np
import pandas as pd

from learn2rank.utils import set_machine


def get_metric(df, metric_name):
    metric = None
    if df.query("name == '{}'".format(metric_name)).shape[0]:
        values = df[df['name'] == metric_name]['value'].values
        if None not in values:
            metric = np.mean(values)

    return metric


@hydra.main(version_base='1.2', config_path='../config', config_name='find_best_model.yaml')
def main(cfg):
    set_machine(cfg)
    output_path = Path(cfg.output_dir)
    Path(cfg.summary_path).mkdir(exist_ok=True, parents=True)
    task_modelName = set()
    result_summary = []
    for result_path in output_path.rglob('results_*.pkl'):
        model_id = '_'.join(result_path.stem.split('_')[1:])
        result = pkl.load(open(result_path, 'rb'))

        task_modelName.add((result_path.parent.stem, result['model_name']))
        ranking_tr = result['train']['ranking']
        ranking_val = result['val']['ranking']

        result_tr = pd.DataFrame(ranking_tr, columns=['id', 'name', 'value'])
        result_val = pd.DataFrame(ranking_val, columns=['id', 'name', 'value'])

        _row = []
        _row.append(result['task'])
        _row.append(result['model_name'])
        _row.append(result_path.parent.stem)
        _row.append(get_metric(result_val, 'kendall-coeff'))
        _row.append(get_metric(result_val, 'spearman-coeff'))
        _row.append(get_metric(result_val, 'top_10_same'))
        _row.append(get_metric(result_val, 'top_10_common'))
        _row.append(get_metric(result_val, 'top_10_penalty'))
        _row.append(result['val']['learning']['mse']
                    if 'learning' in result['val'] and 'mse' in result['val']['learning']
                    else None)
        _row.append(result['val']['learning']['mae']
                    if 'learning' in result['val'] and 'mae' in result['val']['learning']
                    else None)
        _row.append(result['val']['learning']['r2']
                    if 'learning' in result['val'] and 'r2' in result['val']['learning']
                    else None)
        _row.append(get_metric(result_tr, 'kendall-coeff'))
        _row.append(get_metric(result_tr, 'spearman-coeff'))
        _row.append(get_metric(result_tr, 'top_10_same'))
        _row.append(get_metric(result_tr, 'top_10_common'))
        _row.append(get_metric(result_tr, 'top_10_penalty'))
        _row.append(result['train']['learning']['mse']
                    if 'learning' in result['train'] and 'mse' in result['train']['learning']
                    else None)
        _row.append(result['train']['learning']['mae']
                    if 'learning' in result['train'] and 'mae' in result['train']['learning']
                    else None)
        _row.append(result['train']['learning']['r2']
                    if 'learning' in result['train'] and 'r2' in result['train']['learning']
                    else None)
        _row.append(result['model_params'])
        _row.append(model_id)
        result_summary.append(_row)

    # Create summary data frame
    summary_df = pd.DataFrame(result_summary, columns=[
        'task',
        'model_name',
        'size',
        'k-tau_val',
        's-rho_val',
        'top_10_same_val',
        'top_10_common_val',
        'top_10_penalty_val',
        'mse_val',
        'mae_val',
        'r2_val',
        'k-tau',
        's-rho',
        'top_10_same',
        'top_10_common',
        'top_10_penalty',
        'mse',
        'mae',
        'r2',
        'model_params',
        'model_id'
    ])

    summary_df.to_csv(f"{cfg.summary_path}/summary.csv", index=False)

    for task in ['pair_rank_all', 'pair_rank_all_context']:
        name = 'all_context' if task == 'pair_rank_all_context' else 'all'
        summary_path = Path(cfg.summary_path) / f'{name}.csv'

        summary_task = summary_df.query(f"task == '{task}'")
        if summary_task.shape[0]:
            summary_task.to_csv(summary_path, index=False)

            best_model_df = pd.DataFrame(columns=summary_df.columns)
            for mn in set(summary_task['model_name']):
                _df = summary_task[(summary_task.model_name == mn)]
                if _df.shape[0]:
                    best_model_df = pd.concat(
                        [best_model_df,
                         _df[_df[f"{cfg.model_selection}_val"] == _df[f"{cfg.model_selection}_val"].max()]],
                        ignore_index=True)
            best_model_df.to_csv(f'{cfg.summary_path}/best_model_{name}.csv', index=False)

    summary_sizes = summary_df.query(f"task != 'pair_rank_all'")
    summary_sizes = summary_sizes.query("task != 'pair_rank_all_context'")
    for gname, gdf in summary_sizes.groupby('size'):
        summary_path = Path(cfg.summary_path) / f'{gname}.csv'
        gdf.to_csv(summary_path, index=False)

        best_model_df = pd.DataFrame(columns=summary_df.columns)
        for mn in set(gdf.model_name.values):
            _df = gdf[(gdf.model_name == mn)]
            best_model_df = pd.concat(
                [best_model_df,
                 _df[_df[f"{cfg.model_selection}_val"] == _df[f"{cfg.model_selection}_val"].max()]],
                ignore_index=True)
        best_model_df.to_csv(f'{cfg.summary_path}/best_model_{gname}.csv', index=False)


if __name__ == '__main__':
    main()
