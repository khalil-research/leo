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
    reg_lambda = [1, 0.1, 0.001]
    n_estimators = [25, 50, 100, 150, 200]
    max_depth = [3, 5, 7, 10]

    for lam in reg_lambda:
        for md in max_depth:
            for nes in n_estimators:
                lines.append(f'model=GradientBoostingRanker '
                             f'model.max_depth={md} '
                             f'model.n_estimators={nes} '
                             f'model.reg_lambda={lam}')

    return lines


def create_knapsack_table():
    case = 1
    all_lines = []

    lines = create_min_weight_model()
    lines += create_linear_models()
    lines += create_tree_models()
    lines += create_ensemble_models()
    for l in lines:
        all_lines.append(f"{case} python -m learn2rank.scripts.train problem=knapsack {l} task=point_regress")
        case += 1

    lines = create_svm_rank_model()
    for l in lines:
        all_lines.append(f"{case} python -m learn2rank.scripts.train problem=knapsack {l} "
                         f"task=pair_svmrank "
                         f"dataset.path=/home/rahul/Documents/projects/multiobjective_cp2016/resources/datasets/knapsack")
        case += 1

    lines = create_xgb_rank_models()
    for l in lines:
        all_lines.append(f"{case} python -m learn2rank.scripts.train problem=knapsack {l} task=pair_xgbrank "
                         f"dataset.path=/home/rahul/Documents/projects/multiobjective_cp2016/resources/datasets/knapsack")
        case += 1

    return all_lines


def main(args):
    random.seed(args.seed)

    lines = []
    if args.problem == 'knapsack':
        lines = create_knapsack_table()

    lines = "\n".join(lines)
    with open('table.dat', 'w') as fp:
        fp.write(lines)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default='knapsack')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    main(args)
