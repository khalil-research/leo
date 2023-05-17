import pathlib
import pickle as pkl
import shutil
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def main2(cfg):
    # Best params tracking
    best_path = None
    best_loss = np.infty
    best_epoch = 0
    best_cfg = None

    # Iterate and find experiment with best validation loss
    output_path = pathlib.Path(cfg.output_dir)
    for i, f in enumerate(output_path.rglob('*/results.pkl')):
        print(i, f)
        result = pkl.load(open(f, 'rb'))
        cfg = OmegaConf.load(f.parent.joinpath('.hydra/config.yaml'))
        cfg_yaml = OmegaConf.to_yaml(cfg)
        if result['best']['loss']['total'] < best_loss:
            best_path = f
            best_loss = result['best']['loss']['total']
            best_epoch = result['best']['epoch']
            best_cfg = cfg_yaml

    # Copy the best experiment to the pretrained_dir
    shutil.copytree(best_path.parent,
                    f'learn2rank/resources/pretrained/{cfg.pretrained_dir}',
                    dirs_exist_ok=True)

    print(f'Best loss: {best_loss:.4f}')
    print(f'Best epoch: {best_epoch}')
    print(best_cfg)


@hydra.main(version_base='1.1', config_path='../config', config_name='find_best_model.yaml')
def main(cfg):
    output_path = Path(cfg.output_dir)
    Path(cfg.summary_path).mkdir(exist_ok=True, parents=True)
    task_modelName = set()
    result_summary = []
    for result_path in output_path.rglob('results_*.pkl'):
        print(result_path)
        pred_path = '_'.join(result_path.name.split('_')[1:])
        pred_path = result_path.parent / f'prediction_{pred_path}'
        result = pkl.load(open(result_path, 'rb'))

        task_modelName.add((result['task'], result['model_name']))
        ranking_tr = result['tr']['ranking']
        ranking_val = result['val']['ranking']

        result_tr = pd.DataFrame(ranking_tr, columns=['id', 'name', 'value'])
        result_val = pd.DataFrame(ranking_val, columns=['id', 'name', 'value'])

        if result['task'] == 'pair_xgbrank':
            result_summary.append([
                result['task'],
                result['model_name'],
                "_".join(result_path.stem[8:].split("_")[1:]),
                None,
                None,
                None,
                np.mean(result_tr[result_tr['name'] == 'spearman-coeff']['value'].values),
                np.mean(result_tr[result_tr['name'] == 'kendall-coeff']['value'].values),
                np.mean(result_tr[result_tr['name'] == 'top_10_same']['value'].values),
                np.mean(result_tr[result_tr['name'] == 'top_10_common']['value'].values),
                np.mean(result_tr[result_tr['name'] == 'top_10_penalty']['value'].values),
                None,
                None,
                None,
                np.mean(result_val[result_val['name'] == 'spearman-coeff']['value'].values),
                np.mean(result_val[result_val['name'] == 'kendall-coeff']['value'].values),
                np.mean(result_val[result_val['name'] == 'top_10_same']['value'].values),
                np.mean(result_val[result_val['name'] == 'top_10_common']['value'].values),
                np.mean(result_val[result_val['name'] == 'top_10_penalty']['value'].values),
                str(pred_path),
                str(result_path)
            ])
        else:
            result_summary.append([
                result['task'],
                result['model_name'],
                "_".join(result_path.stem[8:].split("_")[1:]),
                result['tr']['learning']['mse'],
                result['tr']['learning']['mae'],
                result['tr']['learning']['r2'],
                np.mean(result_tr[result_tr['name'] == 'spearman-coeff']['value'].values),
                np.mean(result_tr[result_tr['name'] == 'kendall-coeff']['value'].values),
                np.mean(result_tr[result_tr['name'] == 'top_10_same']['value'].values),
                np.mean(result_tr[result_tr['name'] == 'top_10_common']['value'].values),
                np.mean(result_tr[result_tr['name'] == 'top_10_penalty']['value'].values),
                result['val']['learning']['mse'],
                result['val']['learning']['mae'],
                result['val']['learning']['r2'],
                np.mean(result_val[result_val['name'] == 'spearman-coeff']['value'].values),
                np.mean(result_val[result_val['name'] == 'kendall-coeff']['value'].values),
                np.mean(result_val[result_val['name'] == 'top_10_same']['value'].values),
                np.mean(result_val[result_val['name'] == 'top_10_common']['value'].values),
                np.mean(result_val[result_val['name'] == 'top_10_penalty']['value'].values),
                str(pred_path),
                str(result_path)
            ])

    # Create summary data frame
    summary_df = pd.DataFrame(result_summary, columns=[
        'task',
        'model_name',
        'model_params',
        'mse',
        'mae',
        'r2',
        's-rho',
        'k-tau',
        'top_10_same',
        'top_10_common',
        'top_10_penalty',
        'mse_val',
        'mae_val',
        'r2_val',
        's-rho_val',
        'k-tau_val',
        'top_10_same_val',
        'top_10_common_val',
        'top_10_penalty_val',
        'prediction_path',
        'results_path'
    ])
    summary_df.to_csv(f'{cfg.summary_path}/{cfg.problem.size}.csv', index=False)

    best_model_df = pd.DataFrame(columns=summary_df.columns)
    # Select best models
    for task, model_name in task_modelName:
        _df = summary_df[(summary_df.task == task) & (summary_df.model_name == model_name)]
        best_model_df = pd.concat(
            [best_model_df, _df[_df[f"{cfg.model_selection}_val"] == _df[f"{cfg.model_selection}_val"].max()]],
            ignore_index=True)
    best_model_df.to_csv(f'{cfg.summary_path}/best_model_{cfg.problem.size}.csv', index=False)


if __name__ == '__main__':
    main()
