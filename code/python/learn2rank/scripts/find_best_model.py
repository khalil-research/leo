import argparse
import pathlib
import pickle as pkl
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def main2(args):
    # Best params tracking
    best_path = None
    best_loss = np.infty
    best_epoch = 0
    best_cfg = None

    # Iterate and find experiment with best validation loss
    output_path = pathlib.Path(args.output_dir)
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
                    f'learn2rank/resources/pretrained/{args.pretrained_dir}',
                    dirs_exist_ok=True)

    print(f'Best loss: {best_loss:.4f}')
    print(f'Best epoch: {best_epoch}')
    print(best_cfg)


def main(args):
    output_path = Path(args.output_dir)
    result_summary = []
    for result_path in output_path.rglob('results_*.pkl'):
        result = pkl.load(open(result_path, 'rb'))

        ranking_tr = result['tr']['ranking']
        ranking_val = result['val']['ranking']

        result_tr = pd.DataFrame(ranking_tr, columns=['id', 'name', 'value'])
        result_val = pd.DataFrame(ranking_val, columns=['id', 'name', 'value'])

        result_summary.append([
            result_path.stem[8:].split("_")[0],
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
            np.mean(result_val[result_val['name'] == 'top_10_penalty']['value'].values)
        ])

    summary_df = pd.DataFrame(result_summary, columns=[
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
        'mse',
        'mae',
        'r2',
        's-rho',
        'k-tau',
        'top_10_same',
        'top_10_common',
        'top_10_penalty'
    ])
    summary_df.to_csv('model_summary.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='LinearRegression')

    # parser.add_argument('--pretrained_dir', type=str, default='nn__vri')
    args = parser.parse_args()
    main(args)
