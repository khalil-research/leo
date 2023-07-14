import json

import hydra
import pandas as pd

from leo import path

case = 1

min_weight_dict = {'avg_value': 0.0, 'avg_value_by_weight': 0.0, 'max_value': 0.0, 'max_value_by_weight': 0.0,
                   'min_value': 0.0, 'min_value_by_weight': 0.0, 'weight': -1.0}


def create_table_line(case=1, problem='knapsack', n_objs=3, n_vars=60, bin_name='multiobj', mode='SmacI', seed=777,
                      n_jobs=1, cutoff=60, wallclock=300, init_incumbent='min_weight', mem_limit=16,
                      split='train', start_idx=0, n_instances=1):
    return f'{case} python -m leo.label_instance ' \
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
           f'mem_limit={mem_limit} ' \
           f'split={split} ' \
           f'from_pid={start_idx} ' \
           f'num_instances={n_instances} ' \
           f'case={case}\n'


def find_best_run_and_save(cfg, df, split=None):
    if df.shape[0]:
        result_best_run = df.loc[df.groupby('pid').cost.idxmin()].reset_index(drop=True)
        name = f'label_{cfg.problem.size}.csv' if split is None else f'label_{cfg.problem.size}_{split}.csv'
        name = path.label / cfg.problem.name / cfg.problem.size / name

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


@hydra.main(version_base='1.2', config_path='./config', config_name='find_best_label.yaml')
def main(cfg):
    # For data storage
    result_seed_cost, no_traj, no_traj_lines = [], '', ''
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
    smac_run_path = path.SmacI / cfg.problem.name / cfg.problem.size
    for run_path in smac_run_path.rglob(f'run_*'):
        split = run_path.parent.parent.stem
        pid = int(run_path.parent.stem.split('_')[-1])
        run_id = run_path.stem.split('_')[-1]

        traj_path = run_path / 'traj.json'
        # Read traj and save the incumbent
        if traj_path.exists():
            traj_lines = list(traj_path.read_text().strip().split('\n'))
            if len(traj_lines) > 1:
                traj_json = json.loads(traj_lines[-1])
                result_seed_cost.append([pid, run_id, traj_json['cost'], traj_json['incumbent']])
            else:
                traj_json = json.loads(traj_lines[0])
                result_seed_cost.append([pid, run_id, cfg.wallclock, traj_json['incumbent']])
                no_traj_lines += str(run_path) + '\n'
        else:
            # Cases for which traj files don't exist, use min_weight config
            result_seed_cost.append([pid, run_id, cfg.wallclock, min_weight_dict])
            no_traj += str(run_path) + '\n'

    # Run SMAC again for the failed cases
    table_str = create_table(cfg, no_traj)
    table_str = create_table(cfg, no_traj_lines, table_str=table_str)
    label_path = path.label / cfg.problem.name / cfg.problem.size
    label_path.mkdir(exist_ok=True, parents=True)
    label_path.joinpath('no_traj.txt').write_text(no_traj)
    label_path.joinpath('no_traj_lines.txt').write_text(no_traj_lines)
    label_path.joinpath('table.dat').write_text(table_str)

    # All
    df_result_seed = pd.DataFrame(result_seed_cost, columns=['pid', 'seed', 'cost', 'incb'])
    find_best_run_and_save(cfg, df_result_seed)

    # Train
    df_result_seed_train = df_result_seed[df_result_seed['pid'] < cfg.n_train]
    find_best_run_and_save(cfg, df_result_seed_train, split='train')

    # Val
    start, end = cfg.n_train, cfg.n_train + cfg.n_val
    df_result_seed_val = df_result_seed[(df_result_seed['pid'] >= start) & (df_result_seed['pid'] < end)]
    find_best_run_and_save(cfg, df_result_seed_val, split='val')

    # Test
    start, end = cfg.n_train + cfg.n_val, cfg.n_train + cfg.n_val + cfg.n_test
    df_result_seed_test = df_result_seed[(df_result_seed['pid'] >= start) & (df_result_seed['pid'] < end)]
    find_best_run_and_save(cfg, df_result_seed_test, split='test')


if __name__ == '__main__':
    main()
