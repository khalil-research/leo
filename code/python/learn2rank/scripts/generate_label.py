import json
from pathlib import Path

import hydra
import pandas as pd

from learn2rank.utils import set_machine

case = 1

min_weight_dict = {'avg_value': 0.0, 'avg_value_by_weight': 0.0, 'max_value': 0.0, 'max_value_by_weight': 0.0,
                   'min_value': 0.0, 'min_value_by_weight': 0.0, 'weight': -1.0}


def create_table_line(case=1, problem='knapsack', n_objs=3, n_vars=60, bin_name='multiobj', mode='one', seed=777,
                      n_jobs=1, cutoff=60, wallclock=300, init_incumbent='canonical', restore_run=0, new_cutoff=120,
                      new_wallclock=600, mask_mem_limit=0, mem_limit=16, default_width=1.0, label_width=1.0,
                      split='train', start_idx=0, n_instances=1):
    return f'{case} python -m learn2rank.smac_runner ' \
           f'problem={problem} ' \
           f'problem.n_objs={n_objs} ' \
           f'problem.n_vars={n_vars} ' \
           f'bin_name={bin_name} ' \
           f'mode={mode} ' \
           f'seed={seed} ' \
           f'n_jobs={n_jobs} ' \
           f'cutoff_time={cutoff} ' \
           f'wallclock_limit={wallclock} ' \
           f'init_incumbent={init_incumbent} ' \
           f'restore_run={restore_run} ' \
           f'new_cutoff_time={new_cutoff} ' \
           f'new_wallclock_limit={new_wallclock} ' \
           f'mask_mem_limit={mask_mem_limit} ' \
           f'mem_limit={mem_limit} ' \
           f'width.default={default_width} ' \
           f'width.label={label_width} ' \
           f'split={split} ' \
           f'from_pid={start_idx} ' \
           f'num_instances={n_instances} ' \
           f'machine=cc ' \
           f'case={case}\n'


def find_best_run_and_save(cfg, df, split=None):
    if df.shape[0]:
        result_best_run = df.loc[df.groupby('pid').cost.idxmin()].reset_index(drop=True)
        name = f'label_{cfg.problem.size}.csv' if split is None else f'label_{cfg.problem.size}_{split}.csv'
        name = Path(cfg.label_path) / name
        result_best_run.to_csv(name, index=False)


def create_table(cfg, missing_traj, table_str=''):
    if len(missing_traj.strip()):
        global case
        for line in missing_traj.strip().split('\n'):
            blobs = list(line.split('/'))
            run_id = blobs[-1].split('_')[-1]
            pid = blobs[-2].split('_')[-1]
            split = blobs[-3]
            n_objs, n_vars = blobs[-4].split('_')

            table_str += create_table_line(case=case, problem=cfg.problem.name, n_objs=n_objs, n_vars=n_vars,
                                           seed=run_id, cutoff=cfg.cutoff, wallclock=cfg.wallclock,
                                           init_incumbent='min_weight', split=split, start_idx=pid, n_instances=1)
            case += 1

    return table_str


@hydra.main(version_base='1.1', config_path='../config', config_name='generate_label.yaml')
def main(cfg):
    # Path to folder containing smac_runs
    # For example
    # 4_50/             <--- smac_run_path
    #   train/
    #       kp_7_4_50_0/
    #           run_777/
    #   val/
    #       kp_7_4_50_1000/
    #           run_777/
    #   test/
    #       kp_7_4_50_1100/
    #           run_777/
    set_machine(cfg)
    smac_run_path = Path(cfg.smac_run_path)

    # Valid run_ids to use to generate the label file
    valid_run_ids = cfg.run_ids[:cfg.n_run_ids]

    # For data storage
    result_seed_cost = []
    no_traj = ""
    no_traj_lines = ""

    # For all runs
    for vrun_id in valid_run_ids:
        for run_path in smac_run_path.rglob(f'run_{vrun_id}'):
            pid = int(run_path.parent.stem.split("_")[-1])
            split = run_path.parent.parent.stem
            run_id = run_path.stem.split("_")[-1]

            traj_path = run_path / 'traj.json'
            # Read traj and save the incumbent
            if traj_path.exists():
                traj_lines = list(traj_path.read_text().strip().split('\n'))
                if len(traj_lines) > 1:
                    traj_json = json.loads(traj_lines[-1])
                    result_seed_cost.append([pid, run_id, traj_json['cost'], traj_json['incumbent']])
                else:
                    traj_json = json.loads(traj_lines[0])
                    result_seed_cost.append([pid, run_id, 1800, traj_json['incumbent']])
                    no_traj_lines += str(run_path) + "\n"
            else:
                result_seed_cost.append([pid, run_id, 1800, min_weight_dict])
                no_traj += str(run_path) + "\n"

    table_str = create_table(cfg, no_traj)
    table_str = create_table(cfg, no_traj_lines, table_str=table_str)

    Path(cfg.label_path).mkdir(exist_ok=True, parents=True)
    Path(cfg.label_path).joinpath('no_traj.txt').write_text(no_traj)
    Path(cfg.label_path).joinpath('no_traj_lines.txt').write_text(no_traj_lines)
    Path(cfg.label_path).joinpath('table.dat').write_text(table_str)

    # All
    df_result_seed = pd.DataFrame(result_seed_cost, columns=['pid', 'seed', 'cost', 'incb'])
    find_best_run_and_save(cfg, df_result_seed)

    # Train
    df_result_seed_train = df_result_seed[df_result_seed['pid'] < 1000]
    find_best_run_and_save(cfg, df_result_seed_train, split='train')

    # Val
    df_result_seed_val = df_result_seed[(df_result_seed['pid'] >= 1000) & (df_result_seed['pid'] < 1100)]
    find_best_run_and_save(cfg, df_result_seed_val, split='val')

    # Test
    df_result_seed_test = df_result_seed[(df_result_seed['pid'] >= 1100) & (df_result_seed['pid'] < 1200)]
    find_best_run_and_save(cfg, df_result_seed_test, split='test')


if __name__ == '__main__':
    main()
