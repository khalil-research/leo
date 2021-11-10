from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n_instances", type=int, default=250)

    root_path = Path(f'/scratch/rahulpat/knapsack/{seed}')

    obj_var_cfg = ['3_20', '3_40', '3_60', '3_80', '3_100']
    # ['5_20', '7_20', '3_40', '5_40', '7_40', '3_60', '5_60', '3_100']

    for ovc in obj_var_cfg:
        for i in range(args.n_instances):
            f = root_path / ovc / f'train/kp_{args.seed}_{ovc}_{i}.dat'
            entry = " ".join(["timeout 1h",
                              "/home/rahulpat/projects/def-khalile2/rahulpat/multiobjbdd/code/multiobj",
                              str(f), "1", "0", f"./output.$SLURM_JOB_ID"])
            print(entry)
