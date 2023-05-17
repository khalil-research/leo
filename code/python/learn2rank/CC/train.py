import random
from argparse import ArgumentParser


def create_min_weight_model():
    return [f'model=MinWeight']


def create_svm_rank_model():
    lines = []
    for c in [1e-3, 1e-2, 1e-1, 1e-0, 10, 100, 1000, 10000]:
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
    n_estimators = [50, 100, 150]
    max_depth = [5, 7]

    for lam in reg_lambda:
        for md in max_depth:
            for nes in n_estimators:
                lines.append(f'model=GradientBoostingRanker '
                             f'model.max_depth={md} '
                             f'model.n_estimators={nes} '
                             f'model.reg_lambda={lam}')

    return lines


def create_knapsack_table(args):
    case = 1
    all_lines = []
    base = f"python -m learn2rank.scripts.train problem=knapsack problem.n_objs={args.n_objs} problem.n_vars={args.n_vars}"

    lines = create_min_weight_model()
    lines += create_linear_models()
    lines += create_tree_models()
    lines += create_ensemble_models()
    for l in lines:
        all_lines.append(f"{case} {base} {l} task=point_regress")
        case += 1

    lines = create_svm_rank_model()
    for l in lines:
        all_lines.append(f"{case} {base} {l} task=pair_svmrank")
        case += 1

    lines = create_xgb_rank_models()
    for l in lines:
        all_lines.append(f"{case} {base} {l} task=pair_xgbrank")
        case += 1

    return all_lines


def main(args):
    random.seed(args.seed)

    lines = []
    if args.problem == 'knapsack':
        lines = create_knapsack_table(args)

    lines = "\n".join(lines)
    with open(f'train_table_{args.n_objs}_{args.n_vars}.dat', 'w') as fp:
        fp.write(lines)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default='knapsack')
    parser.add_argument('--n_vars', type=int, default=50)
    parser.add_argument('--n_objs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    main(args)
