import copy
import logging
import pickle
import time

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from learn2rank.losses.factory import loss_factory
from .trainer import Trainer

# A logger for this file
log = logging.getLogger(__name__)


class RankingDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index) -> T_co:
        return {'x': self.x[index],
                'y_weight': self.y[index]['pwt'],
                'y_time': self.y[index]['cost'],
                'y_rank': self.y[index]['rank']}


class PyTorchTrainer(Trainer):
    def __init__(self, model=None, data=None, cfg=None):
        super(PyTorchTrainer, self).__init__(data, model, cfg)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Data
        self.tensors = None
        self.loader = self._get_dataloader()

        # Loss function
        self.loss_fn = {'rank': loss_factory.create(self.cfg.model.rp.loss_fn),
                        'weight': loss_factory.create(self.cfg.model.wp.loss_fn),
                        'loss': loss_factory.create(self.cfg.model.tp.loss_fn)}
        self.w = {'rank': 1, 'weight': 0, 'time': 0}

        # Optimizer
        optimizer_cls = getattr(torch.optim, self.cfg.optimizer.name)
        self.optimizer = optimizer_cls(params=self.model.parameters(), lr=self.cfg.optimizer.lr)
        self.model_clone = None

        # initialize wandb
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        config_dict = nesteddict_list2str(config_dict)
        wandb.init(project=self.cfg.wandb.project,
                   mode=self.cfg.wandb.mode,
                   config=config_dict)

        # Result store
        self.rs = self._get_results_store()

    def run(self):
        # Check for checkpoint if available
        # Model clone to hold the best weights during training
        self.model_clone = copy.deepcopy(self.model)
        self.model_clone.eval()

        _time = time.time()
        for epoch in range(self.cfg.run.n_epochs):
            log.info(f"** Epoch {epoch + 1}/{self.cfg.run.n_epochs}")
            self._train_epoch()
            self._val_epoch(epoch)

        _time = time.time() - _time
        self.rs['time']['train'] = _time

        self._save_model(epoch)
        self._save_results()
        wandb.run.summary['best_loss_total'] = self.rs['best']['loss']['total']
        wandb.run.summary['best_loss_rank'] = self.rs['best']['loss']['rank']
        wandb.run.summary['best_loss_rank'] = self.rs['best']['loss']['weight']
        wandb.run.summary['best_loss_rank'] = self.rs['best']['loss']['time']
        wandb.run.summary['best_epoch'] = self.rs['best']['epoch']
        wandb.finish()

        log.info('  Finished Training')
        log.info(f'  Neural Network train time: {_time:.4f}')
        log.info(f"  Best validation loss: {self.rs['best']['loss']['total']}")
        log.info(f"  Best validation epoch: {self.rs['best']['epoch']}")

    def predict(self, split='test'):
        _time = time.time()
        self._val_epoch(0, split='test')
        _time = time.time() - _time

        self.rs['time']['test'] = _time
        self._save_results()

    def _save_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'model': self.model_clone,
            'optimizer_state_dict': self.optimizer.state_dict()
        }, 'final.ckpt')

    def _save_results(self):
        with open('./results.pkl', 'wb') as p:
            pickle.dump(self.rs, p)

    def _train_epoch(self):
        """ Train the model """

        # Put the model in training mode
        self.model.train()
        ep_loss = {'total': [], 'rank': [], 'weight': [], 'time': []}
        for i, data in enumerate(self.loader['tr']):
            # Fetch data
            x, y_rank, y_weight, y_time = data['x'], data['y_rank'], data['y_weight'], data['y_time'],

            # Predict and compute loss
            yp_rank, yp_weight, yp_time = self.model(x)
            loss_dict = self._get_loss(y_rank, y_weight, y_time,
                                       yp_rank=yp_rank, yp_weight=yp_weight, yp_time=yp_time)

            # Backprop
            self.optimizer.zero_grad()
            loss_dict['total'].backward()
            self.optimizer.step()

            # Record losses
            for k in ep_loss.keys():
                ep_loss[k].append(loss_dict[k].detach().cpu().item())

        # Compute epoch-level loss values and record
        for k in ep_loss.keys():
            ep_loss[k] = np.sum(ep_loss[k]) / len(self.loader['tr'])
        self.rs['tr']['loss'].append(ep_loss)
        wandb.log({'tr_total': self.rs['tr']['loss'][-1]['total'],
                   'tr_rank': self.rs['tr']['loss'][-1]['rank'],
                   'tr_weight': self.rs['tr']['loss'][-1]['weight'],
                   'tr_time': self.rs['tr']['loss'][-1]['time']})

        log.info(f"\tTrain loss: {self.rs['tr']['loss'][-1]['total']:.4f}")

    def _val_epoch(self, epoch, split='val'):
        """ Validate model """
        assert split in ['val', 'test']

        # Put the model in eval model. Necessary when using dropout
        self.model.eval()
        with torch.no_grad():
            # Predict on the validation set
            for data in self.loader[split]:
                x, y_weight, y_time, y_rank = data['x'], data['y_weight'], data['y_time'], data['y_rank']
            yp_rank, yp_weight, yp_time = self.model(x)

        # Compute weighted loss
        loss_dict = self._get_loss(y_rank, y_weight, y_time,
                                   yp_rank=yp_rank, yp_weight=yp_weight, yp_time=yp_time,
                                   no_grad=True)

        self.rs[split]['loss'].append(loss_dict)
        # Update best. Only applicable for val split
        if split == 'val' and loss_dict['total'] < self.rs['best']['loss']['total']:
            log.info(f"*** Better model found: epoch {epoch}, old loss {self.rs['best']['loss']['total']:.4f}, "
                     f"new loss {loss_dict['total']:.4f}")
            self.rs['best']['epoch'] = epoch
            self.rs['best']['loss'] = loss_dict
            self.model_clone.load_state_dict(self.model.state_dict())

        # Log
        wandb.log({f'{split}_total': self.rs[split]['loss'][-1]['total'],
                   f'{split}_rank': self.rs[split]['loss'][-1]['rank'],
                   f'{split}_weight': self.rs[split]['loss'][-1]['weight'],
                   f'{split}_time': self.rs[split]['loss'][-1]['time']})
        log.info(f"\t{split} loss: {self.rs[split]['loss'][-1]['total']:.4f}")

    def _get_split_data(self, split='train'):
        x, y = [], []
        for _, v in self.data[split].items():
            _x, _y = v['x'], v['y'][-1]

            _feat = np.hstack((_x['var'], _x['vrank'], _x['inst']))
            _feat = torch.from_numpy(_feat).float()
            _feat = _feat.cuda() if torch.cuda.is_available() else _feat
            x.append(_feat)

            for _yk, _yv in _y.items():
                _y[_yk] = torch.from_numpy(_yv).float()
            _y = {_yk: _yv.cuda() if torch.cuda.is_available() else _yv
                  for _yk, _yv in _y.items()}
            y.append(_y)

        return x, y

    def _get_dataloader(self):
        x_tr, y_tr = self._get_split_data('train')
        x_val, y_val = self._get_split_data('val')
        x_test, y_test = self._get_split_data('test')
        self.tensors = {'x_tr': x_tr, 'y_tr': y_tr,
                        'x_val': x_val, 'y_val': y_val,
                        'x_test': x_test, 'y_test': y_test}

        dataset_tr = RankingDataset(x_tr, y_tr)
        dataset_val = RankingDataset(x_val, y_val)
        loader = {'tr': DataLoader(dataset_tr, batch_size=self.cfg.run.batch_size, shuffle=True),
                  'val': DataLoader(dataset_val, batch_size=len(self.tensors['x_val']), shuffle=False)}

        return loader

    def _get_loss(self, y_rank, y_weight, y_time, yp_rank=None, yp_weight=None, yp_time=None, no_grad=False):
        """ Get loss for different tasks """
        loss_dict = {'rank': torch.zeros(1).to(self.device),
                     'weight': torch.zeros(1).to(self.device),
                     'time': torch.zeros(1).to(self.device),
                     'total': None}

        if yp_rank is not None:
            loss_dict['rank'] = self.loss_fn['rank'].compute(yp_rank, y_rank)
        if yp_weight is not None:
            loss_dict['weight'] = self.loss_fn['weight'].compute(yp_weight, y_weight)
        if yp_time is not None:
            loss_dict['time'] = self.loss_fn['time'].compute(yp_time, y_time)

        if no_grad:
            loss_dict = {k: v.cpu().item() if v is not None else None
                         for k, v in loss_dict.items()}

        loss_dict['total'] = self.w['rank'] * loss_dict['rank'] + \
                             self.w['weight'] * loss_dict['weight'] + \
                             self.w['time'] * loss_dict['time']

        return loss_dict


def nesteddict_list2str(d):
    for k, v in d.items():
        if type(v) == dict:
            nesteddict_list2str(v)
        elif type(v) == list:
            d[k] = ",".join(map(str, v))

    return d
