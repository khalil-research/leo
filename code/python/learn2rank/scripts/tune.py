from subprocess import Popen, PIPE

import hydra
import optuna
from omegaconf import DictConfig


class XGBObjective:
    def __init__(self, machine, time_limit):
        self.machine = machine
        self.time_limit = time_limit

    def __call__(self, trial):
        model_options = f'model=GradientBoostingRanker '

        # Number of gradient boosted trees. Equivalent to number of boosting rounds.
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=25)
        model_options += f'model.n_estimators={n_estimators} '

        # Boosting learning rate (xgb's "eta")
        learning_rate = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        model_options += f'model.learning_rate={learning_rate} '

        # (min_split_loss) Minimum loss reduction required to make a further partition on a leaf node of the tree.
        gamma = trial.suggest_int("gamma", 0, 5)
        model_options += f'model.gamma={gamma} '

        # Maximum tree depth for base learners.
        max_depth = trial.suggest_int("max_depth", 3, 9, step=2)
        model_options += f'model.max_depth={max_depth} '

        # Minimum sum of instance weight (hessian) needed in a child.
        min_child_weight = trial.suggest_int("min_child_weight", 0, 5)
        model_options += f'model.min_child_weight={min_child_weight} '

        # sampling ratio for training data.
        subsample = trial.suggest_float("subsample", 0.2, 1.0)
        model_options += f'model.subsample={subsample} '

        # sampling according to each tree.
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 1.0)
        model_options += f'model.colsample_bytree={colsample_bytree} '

        # L2 regularization weight.
        reg_lambda = trial.suggest_float("lambda", 1e-8, 1.0, log=True)
        model_options += f'model.reg_lambda={reg_lambda} '

        # L1 regularization weight.
        # reg_alpha = trial.suggest_float("alpha", 1e-8, 1.0, log=True)

        grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        model_options += f'model.grow_policy={grow_policy}'

        cmd = "python -m learn2rank.scripts.train mode=TUNE model.verbosity=1 "
        cmd += f'machine={self.machine} '
        cmd += model_options
        print(cmd)
        io = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE)

        # Call target algorithm with cutoff time
        (stdout_, stderr_) = io.communicate(self.time_limit)
        stdout, stderr = stdout_.decode('utf-8'), stderr_.decode('utf-8')
        val_tau = float(stdout.strip().split('val_tau:')[1].strip())

        return val_tau


@hydra.main(version_base='1.2', config_path='../config', config_name='tune.yaml')
def main(cfg: DictConfig):
    study = optuna.create_study(direction='maximize')
    study.optimize(globals()[cfg.objective](cfg.machine,
                                            cfg.time_limit),
                   n_trials=cfg.n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    main()
