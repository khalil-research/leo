import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path
from subprocess import Popen, PIPE

CURR_FILE_PATH = Path(__file__)
RES_PATH = {
    'cc': Path('/home/rahulpat/scratch/l2o_resources'),
    'desktop': '/home/rahul/Documents/projects/multiobjective-bdd-vo/python/learn2rank/resources',
    'laptop': CURR_FILE_PATH.parent.parent / 'resources'
}


def preprocess_setcover(res_path, inst_path):
    binary = res_path / 'min_bandwidth'
    cmd = f"{binary} {inst_path} 4 1"

    # Maximal virtual memory for subprocesses (in bytes).
    io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE)

    # Call target algorithm with cutoff time
    (stdout_, stderr_) = io.communicate()

    # Decode and parse output
    stdout, stderr = stdout_.decode('utf-8'), stderr_.decode('utf-8')
    stdout = stdout.strip()
    lines = list(stdout.split('\n'))

    inst_lines = lines[1:-1]
    inst = "\n".join(inst_lines)

    _time = float(lines[-1])

    return inst, _time


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--machine', type=str, default='laptop')
    parser.add_argument('--size', type=str, default='100_3')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--from_pid', type=int, default=0)
    parser.add_argument('--to_pid', type=int, default=1)
    args = parser.parse_args()

    SETCOVER_PATH = RES_PATH[args.machine] / 'instances/setcovering'
    split_path = SETCOVER_PATH / args.size / args.split
    times = {}
    for pid in range(args.from_pid, args.to_pid):
        instance, _time = preprocess_setcover(RES_PATH[args.machine],
                                              split_path / f'bp_7_{args.size}_{pid}.dat')
        out_path = RES_PATH[args.machine] / 'instances/setcovering_preprocessed'
        out_path = out_path / f'{args.size}/{args.split}'
        out_path = out_path / f'bp_7_{args.size}_{pid}.dat'

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.open('w').write(instance)

        times[f'{args.size}/{args.split}/{pid}'] = _time

    pkl.dump(times,
             open(f'sc_pp_time_{args.size}_{args.split}_{args.from_pid}_{args.to_pid}.pkl',
                  'wb'))
