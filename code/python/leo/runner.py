import argparse
import os


def main(args):
    with open(args.t, 'r') as fp:
        lines = list(fp.readlines())

        for line in lines:
            cmd = line.strip()
            os.system(" ".join(cmd.strip().split(' ')[1:]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, required=True)

    args = parser.parse_args()
    main(args)
