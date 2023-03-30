import random
from argparse import ArgumentParser


def create_table_line():
    pass


def create_linear_models(task, weights=0):
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


def create_tree_models(task):
    lines = []
    max_features = ["sqrt", "log2"]
    max_depth = [3, 5, 10]

    # Default model
    lines.append(f'model=DecisionTreeRegressor')

    # Grid search
    for mf in max_features:
        for md in max_depth:
            lines.append(f'model=DecisionTreeRegressor model.max_depth={md} model.max_features={mf}')

    return lines


def create_ensemble_models(task):
    lines = []
    max_features = ["sqrt", "log2"]
    n_estimators = [25, 50]
    max_depth = [3, 5]

    lines.append(f'model=DecisionTreeRegressor')
    for mf in max_features:
        for md in max_depth:
            for nes in n_estimators:
                lines.append(f'model=GradientBoostingRegressor '
                             f'model.max_depth={md} '
                             f'model.max_features={mf} '
                             f'model.n_estimators={nes}')

    return lines


def create_knapsack_table(task):
    lines = create_linear_models(task)
    lines += create_tree_models(task)
    lines += create_ensemble_models(task)

    case = 1
    lines1 = []
    for l in lines:
        lines1.append(f"{case} python -m learn2rank.scripts.train problem=knapsack {l}")
        case += 1

    return lines1


def main(args):
    random.seed(args.seed)

    lines = []
    if args.problem == 'knapsack':
        lines = create_knapsack_table(args.task)

    lines = "\n".join(lines)
    with open('table.dat', 'w') as fp:
        fp.write(lines)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default='knapsack')
    parser.add_argument('--task', type=str, default='point_regress')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    main(args)
