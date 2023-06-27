import random
from argparse import ArgumentParser


def test_knapsack_table():
    sizes = ['5_40', '6_40', '7_40', '4_50', '3_60', '3_70', '3_80']
    case = 1
    all_lines = []

    for size in sizes:
        n_objs, n_vars = size.split('_')
        base = f"python -m learn2rank.scripts.test problem=knapsack problem.n_objs={n_objs} problem.n_vars={n_vars} " \
               f"mode=all task=point_regress dataset.fused=0"
        all_lines.append(f"{case} {base}")
        case += 1

        base = f"python -m learn2rank.scripts.test problem=knapsack problem.n_objs={n_objs} problem.n_vars={n_vars} " \
               f"mode=all task=pair_rank dataset.fused=0"
        all_lines.append(f"{case} {base}")
        case += 1

    base = f"python -m learn2rank.scripts.test problem=knapsack mode=all task=pair_rank_all dataset.fused=1"
    all_lines.append(f"{case} {base}")
    case += 1

    base = f"python -m learn2rank.scripts.test problem=knapsack mode=all task=pair_rank_all_context dataset.fused=1"
    all_lines.append(f"{case} {base}")
    case += 1

    return all_lines


def main(args):
    random.seed(args.seed)

    lines = []
    if args.problem == 'knapsack':
        lines = test_knapsack_table()

    lines = "\n".join(lines)
    with open(f'test_table.dat', 'w') as fp:
        fp.write(lines)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default='knapsack')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    main(args)
