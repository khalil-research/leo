import pickle as pkl

import hydra
import numpy as np
import pandas as pd

from leo import path


def get_metric(df, metric_name):
    metric = None
    if df.query("name == '{}'".format(metric_name)).shape[0]:
        values = df[df['name'] == metric_name]['value'].values
        if None not in values:
            metric = np.mean(values)

    return metric


def get_dataset_config(dataset_name):
    blobs = dataset_name.split('_')
    has_context = True if 'context' in blobs else False
    is_fused = True if 'all' in blobs else False

    size = ''
    if not is_fused:
        size = f'{blobs[0]}_{blobs[1]}'

    return {'size': size, 'context': has_context, 'fused': is_fused}


@hydra.main(version_base='1.2', config_path='./config', config_name='find_best_model.yaml')
def main(cfg):
    path.model_summary.joinpath(cfg.problem.name).mkdir(exist_ok=True, parents=True)

    result_summary = []
    for result_path in path.prediction.rglob('results_*.pkl'):
        model_id = '_'.join(result_path.stem.split('_')[1:])
        dataset_name = result_path.parent.stem
        dataset_cfg = get_dataset_config(dataset_name)

        result = pkl.load(open(result_path, 'rb'))
        ranking_tr = result['train']['ranking']
        ranking_val = result['val']['ranking']
        result_tr = pd.DataFrame(ranking_tr, columns=['id', 'name', 'value'])
        result_val = pd.DataFrame(ranking_val, columns=['id', 'name', 'value'])

        _row = []
        _row.append(result['task'])
        _row.append(result['model_name'])
        _row.append(dataset_name)
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
        _row.append(dataset_cfg['size'])
        _row.append(dataset_cfg['context'])
        _row.append(dataset_cfg['fused'])
        result_summary.append(_row)

    # Create summary data frame
    summary_df = pd.DataFrame(result_summary, columns=[
        'task',
        'model_name',
        'dataset_name',
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
        'model_id',
        'size',
        'context',
        'fused'
    ])
    summary_df.to_csv(f"{path.model_summary / cfg.problem.name}/summary.csv", index=False)

    for dname, ddf in summary_df.groupby('dataset_name'):
        summary_path = path.model_summary / cfg.problem.name / f'{dname}.csv'
        ddf.to_csv(summary_path, index=False)

        best_model_df = pd.DataFrame(columns=summary_df.columns)
        for mn in set(ddf.model_name.values):
            _df = ddf[(ddf.model_name == mn)]
            best_model_df = pd.concat(
                [best_model_df,
                 _df[_df[f"{cfg.model_selection}_val"] == _df[f"{cfg.model_selection}_val"].max()]],
                ignore_index=True)
        best_model_df.to_csv(f'{path.model_summary / cfg.problem.name}/best_{dname}.csv', index=False)


if __name__ == '__main__':
    main()
