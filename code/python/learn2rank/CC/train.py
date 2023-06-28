import random
from argparse import ArgumentParser


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
    for c in [1e-3, 1e-2, 1e-1, 1e-0, 10, 100]:
        lines.append(f'model=SVMRank model.c={c}')

    return lines


def create_linear_models(weights=0):
    # Linear Regression model
    lines = [f'model=LinearRegression model.weights={weights}']

    # Default models
    lines.append(f'model=Lasso')
    lines.append(f'model=Ridge')
    # Grid search
    alphas = [0.1, 0.001, 0.0001, 1, 5, 10]
    for alpha in alphas:
        lines.append(f'model=Lasso model.alpha={alpha} model.weights={weights}')
        lines.append(f'model=Ridge model.alpha={alpha} model.weights={weights}')

    return lines


def create_tree_models():
    lines = []
    max_features = ["auto", "log2"]
    max_depth = [3, 5, 10]

    # Grid search
    for mf in max_features:
        for md in max_depth:
            lines.append(f'model=DecisionTreeRegressor model.max_depth={md} model.max_features={mf}')

    return lines


def create_ensemble_models():
    lines = []
    max_features = ["auto", "log2"]
    n_estimators = [25, 50, 100]
    max_depth = [3, 5]

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
    reg_lambda = [0.1, 0.01]
    n_estimators = [100, 150]
    max_depth = [5, 7]
    lr = [0.1, 0.01]

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


def create_knapsack_table(args):
    sizes = ['5_40', '6_40', '7_40', '4_50', '3_60', '3_70', '3_80']
    case = 1
    all_lines = []

    for size in sizes:
        n_objs, n_vars = size.split('_')
        base = f"python -m learn2rank.scripts.train problem=knapsack problem.n_objs={n_objs} problem.n_vars={n_vars}"

        lines = create_heuristic_order_model()
        lines += create_lex_model()
        lines += create_smac_model()
        lines += create_linear_models()
        lines += create_tree_models()
        lines += create_ensemble_models()
        for l in lines:
            all_lines.append(f"{case} {base} {l} task=point_regress")
            case += 1

        lines = create_svm_rank_model()
        for l in lines:
            all_lines.append(f"{case} {base} {l} task=pair_rank")
            case += 1

        lines = create_xgb_rank_models()
        for l in lines:
            all_lines.append(f"{case} {base} {l} task=pair_rank")
            case += 1

    lines = create_xgb_rank_models()
    base = f"python -m learn2rank.scripts.train problem=knapsack"
    for l in lines:
        all_lines.append(f"{case} {base} {l} task=pair_rank_all dataset.fused=1")
        case += 1

    for l in lines:
        all_lines.append(f"{case} {base} {l} task=pair_rank_all_context dataset.fused=1")
        case += 1

    return all_lines


def main(args):
    random.seed(args.seed)

    lines = []
    if args.problem == 'knapsack':
        lines = create_knapsack_table(args)

    lines = "\n".join(lines)
    with open(f'train_table.dat', 'w') as fp:
        fp.write(lines)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default='knapsack')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    main(args)
