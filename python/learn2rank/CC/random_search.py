import argparse
import random

import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


class ContinuousValueSampler(object):
    """ A class to sample uniformly at random in the range of [lb,ub].
        Additionally includes a probability of sampling zero if needed.  """

    def __init__(self, lb, ub, prob_zero=0.0):
        self.lb = lb
        self.ub = ub
        self.prob_zero = prob_zero

    def sample(self):
        if np.random.rand() < self.prob_zero:
            return 0
        return np.round(np.random.uniform(self.lb, self.ub), 5)


class DiscreteSampler(object):
    """ A class to sample uniformly at random in the range of [lb,ub]. """

    def __init__(self, choices):
        self.choices = choices

    def sample(self):
        return np.random.choice(self.choices)


def get_basic_config():
    LR_LB, LR_UB = 1e-4, 1e-2
    L1_LB, L1_UB = 1e-5, 1e-1
    L2_LB, L2_UB = 1e-5, 1e-1
    L1_ZERO, L2_ZERO = 0.25, 0.25

    config = {
        'batch_size': DiscreteSampler([32, 64, 128]),
        'lr': ContinuousValueSampler(LR_LB, LR_UB),
        'n_epochs': DiscreteSampler([1000]),
        'dropout': ContinuousValueSampler(0.0, 0.5),
    }

    return config


def get_nn_config():
    config = get_basic_config()
    config.update(
        {
            'inp': DiscreteSampler([37]),
            'out': DiscreteSampler([64, 128]),
            'layers': DiscreteSampler([None,
                                       '128', '256', '512',
                                       '128,128', '256,256', '512,512'])
        }
    )

    return config


def get_te_config():
    config = get_basic_config()
    config.update(
        {
            'tfe_n_heads': DiscreteSampler([3, 4, 5, 6, 7, 8]),
            'tfe_dp': ContinuousValueSampler(0.1, 0.6, prob_zero=0.1),
            'tfe_n_layers': DiscreteSampler([2, 3, 4, 5, 6]),
            'tfe_act': DiscreteSampler(['gelu', 'relu'])
        }
    )

    return config


def get_config(model_type):
    """ Gets the config for the given model_type. """
    if model_type == "nn":
        return get_nn_config()
    elif model_type == "te":
        return get_te_config()
    else:
        raise Exception(f"Config not defined for model_type [{model_type}]")


def sample_config(model_type, config):
    """ Samples a confiuration for nn_single_cut. """
    config_cmd = f"python -m learn2rank.scripts.train_model " \
                 f"wandb.mode=offline " \
                 f"run.n_epochs={config['n_epochs'].sample()} " \
                 f"run.batch_size={config['batch_size'].sample()} " \
                 f"optimizer.lr={config['lr'].sample()} "

    if model_type == 'nn':
        layers = config['layers'].sample()
        layers = '' if layers is None else layers
        layers = '\\\'' + layers + '\\\''
        config_cmd += f"model.tf_enc.switch=off " \
                      f"model.feat_enc.inp={config['inp'].sample()} " \
                      f"model.feat_enc.out={config['out'].sample()} " \
                      f"model.feat_enc.layers={layers} " \
                      f"model.feat_enc.dp={config['dropout'].sample()} " \
                      f"model.feat_enc.dp_last=on"

    if model_type == 'te':
        config_cmd += f"model.tf_enc.n_heads={config['tfe_n_heads'].sample()} " \
                      f"model.tf_enc.dp={config['tfe_dp'].sample()} " \
                      f"model.tf_enc.act={config['tfe_act'].sample()} " \
                      f"model.tf_enc.n_layers={config['tfe_n_layers'].sample()}"

    return config_cmd


def main(args):
    set_seed(1501)
    cmds = []

    config = get_config(args.model_type)
    for _ in range(args.n_configs):
        cmds.append(sample_config(args.model_type, config))

    # write to text file
    textfile = open(args.file_name, "w")
    for i, cmd in enumerate(cmds):
        textfile.write(f"{i + 1} {cmd} case={i + 1}\n")
    textfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a list of configs to run for random search.')
    parser.add_argument('--model_type', type=str, default='nn')
    parser.add_argument('--n_configs', type=int, default=1000)
    parser.add_argument('--file_name', type=str, default='table.dat')
    args = parser.parse_args()

    main(args)
