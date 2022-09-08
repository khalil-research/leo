import copy
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


class RankingDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index) -> T_co:
        return {'x': self.x[index],
                'y_weight': self.y[index]['fwt'],
                'y_time': self.y[index]['cost'],
                'y_rank': self.y[index]['rank']}


class PyTorchTrainer(Trainer):
    def __init__(self, model=None, data=None, cfg=None):
        super(PyTorchTrainer, self).__init__(data, model, cfg)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.model_clone = copy.deepcopy(self.model)
        self.model_clone.eval()

        # initialize wandb
        wandb.init(**self.cfg.wandb,
                   config=OmegaConf.to_container(self.cfg, resolve=True))

        # Result store
        self.rs = self._get_results_store()

    def run(self):
        # Check for checkpoint if available

        _time = time.time()
        for epoch in range(self.cfg.run.n_epochs):
            print(f"** Epoch {epoch + 1}/{self.cfg.run.n_epochs}")
            self._train_epoch()
            self._val_epoch(epoch)

        _time = time.time() - _time
        self.rs['time']['train'] = _time

        self._save(epoch)

        print('  Finished Training')
        print('  Neural Network train time:', _time)

    def predict(self, split='test'):
        pass
        # self.model.eval()
        # with torch.no_grad:
        #     yp_rank, yp_weight, yp_time = self.model(x)

    def _save(self, epoch):
        self._save_model(epoch)
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

        print(f"\tTrain loss: {self.rs['tr']['loss'][-1]['total']}")

    def _val_epoch(self, epoch, split='val'):
        """ Validate model """

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

        # for k in loss_dict.keys():
        #     print(k, loss_dict[k], type(loss_dict[k]))
        #     loss_dict[k] = loss_dict[k].numpy()
        self.rs[split]['loss'].append(loss_dict)
        wandb.log({f'{split}_total': self.rs[split]['loss'][-1]['total'],
                   f'{split}_rank': self.rs[split]['loss'][-1]['rank'],
                   f'{split}_weight': self.rs[split]['loss'][-1]['weight'],
                   f'{split}_time': self.rs[split]['loss'][-1]['time']})

        print(f"\tVal loss: {self.rs['val']['loss'][-1]['total']}")
        # Save best
        if split == 'val' and loss_dict['total'] < self.rs['best']['loss']['total']:
            print(f"*** Better model found: epoch {epoch}, old loss {self.rs['best']['loss']['total']}, "
                  f"new loss {loss_dict['total']}")
            self.rs['best']['epoch'] = epoch
            self.rs['best']['loss'] = loss_dict
            self.model_clone.load_state_dict(self.model.state_dict())

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
        loss_dict = {'rank': torch.zeros(1), 'weight': torch.zeros(1), 'time': torch.zeros(1), 'total': torch.zeros(1)}
        if yp_rank is not None:
            loss_dict['rank'] = self.loss_fn['rank'].compute(yp_rank, y_rank)
        if yp_weight is not None:
            loss_dict['weight'] = self.loss_fn['weight'].compute(yp_weight, y_weight)
        if yp_time is not None:
            loss_dict['time'] = self.loss_fn['time'].compute(yp_time, y_time)

        if no_grad:
            for k in loss_dict.keys():
                loss_dict[k] = loss_dict[k].cpu().item()

        loss_dict['total'] = self.w['rank'] * loss_dict['rank'] + \
                             self.w['weight'] * loss_dict['weight'] + \
                             self.w['time'] * loss_dict['time']

        return loss_dict

    @staticmethod
    def _get_results_store():
        return {
            'tr': {
                'loss': []
            },
            'val': {
                'loss': []
            },
            'test': {
                'loss': []
            },
            'best': {
                'loss': {'total': np.infty, 'rank': np.infty, 'weight': np.infty, 'time': np.infty},
                'epoch': None
            },
            'time': {
                'train': 0.0,
                'test': 0.0,
                'eval': 0.0
            }
        }
