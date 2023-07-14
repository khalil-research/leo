from pathlib import Path

import hydra

case = 1

# In hours
HOUR2SEC = 60 * 60
JOB_TIME = 3 * HOUR2SEC

# Cutoff and Wallclock presets
# We set cutoff equal to wallclock to avoid early termination
BASE_CUTOFF = BASE_WALLCLOCK = 5 * 60
BASE_CUTOFF_2 = BASE_WALLCLOCK_2 = 2 * BASE_WALLCLOCK
BASE_CUTOFF_3 = BASE_WALLCLOCK_3 = 3 * BASE_WALLCLOCK
BASE_CUTOFF_4 = BASE_WALLCLOCK_4 = 4 * BASE_WALLCLOCK
BASE_CUTOFF_ALL = BASE_WALLCLOCK_ALL = 12 * HOUR2SEC

sizes = [(5, 40), (6, 40), (7, 40), (4, 50), (3, 60), (3, 70), (3, 80)]


def create_table_line_smac(case=0, problem='knapsack', n_objs=3, n_vars=60, bin_name='multiobj', mode='SmacI', seed=777,
                           n_jobs=1, cutoff=60, wallclock=300, init_incumbent='min_weight', mem_limit=16, split='train',
                           start_idx=0, n_instances=1):
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


def get_smac_inst_cmd():
    global case

    seeds = [777]
    configs = {
        (5, 40): (BASE_CUTOFF, BASE_WALLCLOCK),
        (4, 50): (BASE_CUTOFF, BASE_WALLCLOCK),
        (3, 60): (BASE_CUTOFF, BASE_WALLCLOCK),
        (6, 40): (BASE_CUTOFF_4, BASE_WALLCLOCK_4),
        (7, 40): (BASE_CUTOFF_4, BASE_WALLCLOCK_4),
        (3, 70): (BASE_CUTOFF_4, BASE_WALLCLOCK_4),
        (3, 80): (BASE_CUTOFF_4, BASE_WALLCLOCK_4)
    }
    table_str = ''
    # Uncomment below line for all splits
    # splits = [('train', 0, 1000), ('val', 1000, 1100)]
    splits = [('train', 0, 1)]

    # Comment below line for sizes
    sizes = [(3, 60)]

    for split in splits:
        key, start, end = split
        for s in sizes:
            cfg = configs[s]
            for seed in seeds:
                for pid in range(start, end):
                    table_str += create_table_line_smac(case=case, problem='knapsack', n_objs=s[0], n_vars=s[1],
                                                        split=key, start_idx=pid, n_instances=1,
                                                        cutoff=cfg[0], wallclock=cfg[1],
                                                        mode='SmacI', seed=seed)
                    case += 1

    return table_str


def get_smac_dataset_cmd():
    global case

    # Comment below line for sizes
    sizes = [(3, 60)]

    seeds = [777]
    table_str = ''

    for s in sizes:
        for seed in seeds:
            table_str += create_table_line_smac(case=case, problem='knapsack', n_objs=s[0], n_vars=s[1],
                                                split='train', start_idx=0, n_instances=1000,
                                                cutoff=BASE_CUTOFF_ALL, wallclock=BASE_WALLCLOCK_ALL,
                                                mode='SmacD', seed=seed)
            case += 1

    return table_str


def get_best_label_cmd():
    global case

    table_str = ''
    for s in sizes:
        table_str += f'{case} python -m leo.find_best_label problem.n_objs={s[0]} problem.n_vars={s[1]}\n'
        case += 1

    return table_str


def get_dataset_cmd():
    global case

    table_str = ''
    table_str += f'{case} python -m leo.generate_dataset task=point_regress fused=0 context=1\n'
    case += 1

    table_str += f'{case} python -m leo.generate_dataset task=pair_rank fused=0 context=0\n'
    case += 1

    table_str += f'{case} python -m leo.generate_dataset task=pair_rank fused=0 context=1\n'
    case += 1

    table_str += f'{case} python -m leo.generate_dataset task=pair_rank fused=1 context=0\n'
    case += 1

    table_str += f'{case} python -m leo.generate_dataset task=pair_rank fused=1 context=1\n'
    case += 1

    return table_str


def get_train_cmd():
    def create_heuristic_order_model():
        return ['model=HeuristicWeight model.sort=asc',
                'model=HeuristicWeight model.sort=des',
                'model=HeuristicValue model.agg=mean model.sort=asc',
                'model=HeuristicValue model.agg=mean model.sort=des',
                'model=HeuristicValue model.agg=min model.sort=asc',
                'model=HeuristicValue model.agg=min model.sort=des',
                'model=HeuristicValue model.agg=max model.sort=asc',
                'model=HeuristicValue model.agg=max model.sort=des',
                'model=HeuristicValueByWeight model.agg=mean model.sort=des',
                'model=HeuristicValueByWeight model.agg=max model.sort=des',
                'model=HeuristicValueByWeight model.agg=min model.sort=des',
                'model=Random model.seed=0',
                'model=Random model.seed=1',
                'model=Random model.seed=2',
                'model=Random model.seed=3',
                'model=Random model.seed=4']

    def create_lex_model():
        return [f'model=Lex']

    def create_smac_model():
        return ['model=SmacD',
                'model=SmacI']

    def create_svm_rank_model():
        lines = []
        # for c in [1e-3, 1e-2, 1e-1, 1e-0, 10, 100]:
        #     lines.append(f'model=SVMRank model.c={c}')
        for c in [10]:
            lines.append(f'model=SVMRank model.c={c}')

        return lines

    def create_linear_models(weights=0):
        # Linear Regression model
        lines = [f'model=LinearRegression model.weights={weights}']

        # Default models
        lines.append(f'model=Lasso')
        lines.append(f'model=Ridge')
        # Grid search
        # alphas = [0.1, 0.001, 0.0001, 1, 5, 10]
        alphas = [0.1]
        for alpha in alphas:
            lines.append(f'model=Lasso model.alpha={alpha} model.weights={weights}')
            lines.append(f'model=Ridge model.alpha={alpha} model.weights={weights}')

        return lines

    def create_tree_models():
        lines = []
        # max_features = ["auto", "log2"]
        max_features = ["auto"]
        # max_depth = [3, 5, 10]
        max_depth = [5]

        # Grid search
        for mf in max_features:
            for md in max_depth:
                lines.append(f'model=DecisionTreeRegressor model.max_depth={md} model.max_features={mf}')

        return lines

    def create_ensemble_models():
        lines = []
        # max_features = ["auto", "log2"]
        # n_estimators = [25, 50, 100]
        # max_depth = [3, 5]
        max_features = ["auto"]
        n_estimators = [50]
        max_depth = [5]

        for mf in max_features:
            for md in max_depth:
                for nes in n_estimators:
                    lines.append(f'model=GradientBoostingRegressor '
                                 f'model.max_depth={md} '
                                 f'model.max_features={mf} '
                                 f'model.n_estimators={nes}')

        return lines

    def create_xgb_rank_models():
        lines = []
        # reg_lambda = [0.1, 0.01]
        # n_estimators = [100, 150]
        # max_depth = [5, 7]
        # lr = [0.3, 0.1, 0.01]
        reg_lambda = [0.1]
        n_estimators = [100]
        max_depth = [7]
        lr = [0.3]

        for lam in reg_lambda:
            for md in max_depth:
                for nes in n_estimators:
                    for _lr in lr:
                        lines.append(f'model=GradientBoostingRanker '
                                     f'model.max_depth={md} '
                                     f'model.n_estimators={nes} '
                                     f'model.reg_lambda={lam} '
                                     f'model.learning_rate={_lr}')

        return lines

    global case
    sizes = [(3, 60)]
    table_str = ''

    for s in sizes:
        base = f'python -m leo.train problem=knapsack problem.n_objs={s[0]} problem.n_vars={s[1]}'
        lines = []
        lines += create_heuristic_order_model()
        lines += create_lex_model()
        lines += create_smac_model()
        lines += create_linear_models()
        lines += create_tree_models()
        lines += create_ensemble_models()
        for l in lines:
            table_str += f'{case} {base} {l} task=point_regress fused=0 context=1\n'
            case += 1

        lines = create_svm_rank_model()
        for l in lines:
            table_str += f'{case} {base} {l} task=pair_rank fused=0 context=0\n'
            case += 1

        lines = create_xgb_rank_models()
        for l in lines:
            table_str += f'{case} {base} {l} task=pair_rank fused=0 context=0\n'
            case += 1

            table_str += f'{case} {base} {l} task=pair_rank fused=0 context=1\n'
            case += 1

    lines = create_xgb_rank_models()
    base = f'python -m leo.train problem=knapsack'
    for l in lines:
        table_str += f'{case} {base} {l} task=pair_rank fused=1 context=0\n'
        case += 1

    for l in lines:
        table_str += f'{case} {base} {l} task=pair_rank fused=1 context=1\n'
        case += 1

    return table_str


def get_best_model_cmd():
    global case

    table_str = f'{case} python -m leo.find_best_model problem=knapsack'
    case += 1

    return table_str


def get_test_cmd():
    global case
    table_str = ''

    for s in sizes:
        table_str += f'{case} python -m leo.test problem=knapsack problem.n_objs={s[0]} problem.n_vars={s[1]} ' \
                     f'mode=best task=point_regress fused=0 context=1\n'
        case += 1

        table_str += f'{case} python -m leo.test problem=knapsack problem.n_objs={s[0]} problem.n_vars={s[1]} ' \
                     f'mode=best task=pair_rank fused=0 context=0\n'
        case += 1

        table_str += f'{case} python -m leo.test problem=knapsack problem.n_objs={s[0]} problem.n_vars={s[1]} ' \
                     f'mode=best task=pair_rank fused=0 context=1\n'
        case += 1

    table_str += f'{case} python -m leo.test problem=knapsack mode=best task=pair_rank fused=0 context=0\n'
    case += 1

    table_str += f'{case} python -m leo.test problem=knapsack mode=best task=pair_rank fused=0 context=1\n'
    case += 1

    return table_str


def get_eval_order_cmd():
    global case
    table_str = ''

    for s in sizes:
        table_str += f'{case} python -m leo.eval_order problem=knapsack problem.n_objs={s[0]} problem.n_vars={s[1]} ' \
                     f'mode=best task=pair_rank model=GradientBoostingRanker fused=0 context=0\n'
        case += 1

        table_str += f'{case} python -m leo.eval_order problem=knapsack problem.n_objs={s[0]} problem.n_vars={s[1]} ' \
                     f'mode=best task=pair_rank model=GradientBoostingRanker fused=0 context=1\n'
        case += 1

    table_str += f'{case} python -m leo.eval_order problem=knapsack mode=best task=pair_rank ' \
                 f'model=GradientBoostingRanker fused=1 context=0\n'
    case += 1

    table_str += f'{case} python -m leo.eval_order problem=knapsack mode=best task=pair_rank ' \
                 f'model=GradientBoostingRanker fused=1 context=1\n'
    case += 1

    return table_str


@hydra.main(version_base='1.2', config_path='./config', config_name='get_cmd.yaml')
def main(cfg):
    cmds = ''

    # Phase 1
    if cfg.get_smac_inst_cmd:
        cmds += get_smac_inst_cmd()
    if cfg.get_smac_dataset_cmd:
        cmds += get_smac_dataset_cmd()
    if cfg.get_best_label_cmd:
        cmds += get_best_label_cmd()

    # Phase 2
    if cfg.get_dataset_cmd:
        cmds += get_dataset_cmd()
    if cfg.get_train_cmd:
        cmds += get_train_cmd()

    # Phase 3
    if cfg.get_best_model_cmd:
        cmds += get_best_model_cmd()
    if cfg.get_test_cmd:
        cmds += get_test_cmd()
    if cfg.get_eval_order_cmd:
        cmds += get_eval_order_cmd()

    Path('cmds.txt').write_text(cmds)


if __name__ == '__main__':
    main()
