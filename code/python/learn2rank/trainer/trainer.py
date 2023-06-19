import pickle
from abc import ABC, abstractmethod
from pathlib import Path


class Trainer(ABC):
    def __init__(self, data, model, config, ps, rs):
        self.data = data
        self.model = model
        self.cfg = config
        self.rs = rs
        self.ps = ps

        self.res_path = Path(self.cfg.res_path[self.cfg.machine])
        if self.cfg.dataset.fused and 'context' not in self.cfg.task:
            self.pred_path = self.res_path / f'predictions/{self.cfg.problem.name}/all'
        elif self.cfg.dataset.fused and 'context' in self.cfg.task:
            self.pred_path = self.res_path / f'predictions/{self.cfg.problem.name}/all_context'
        else:
            self.pred_path = self.res_path / f'predictions/{self.cfg.problem.name}/{self.cfg.problem.size}'
        self.pred_path.mkdir(parents=True, exist_ok=True)

        self.val_tau = None

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def _save_predictions(self):
        out_path = self.pred_path / f'prediction_{self.model.id}.pkl'
        with open(out_path, 'wb') as p:
            pickle.dump(self.ps, p)

    def _save_results(self):
        if self.val_tau is not None:
            Path(self.pred_path / f'val_tau_{self.model.id}.txt').write_text(str(self.val_tau))

        out_path = self.pred_path / f'results_{self.model.id}.pkl'
        with open(out_path, 'wb') as p:
            pickle.dump(self.rs, p)

    def _get_preds_store(self):
        return {
            'task': self.cfg.task,
            'model_name': self.cfg.model.name,
            'model_id': self.model.id,
            'model_params': str(self.model),
            'tr': {
                'names': [],
                'n_items': [],
                'score': [],
                'rank': [],
                'order': []
            },
            'val': {
                'names': [],
                'n_items': [],
                'score': [],
                'rank': [],
                'order': []
            },
            'test': {
                'names': [],
                'n_items': [],
                'score': [],
                'rank': [],
                'order': []
            }
        }

    def _get_results_store(self):
        return {
            'task': self.cfg.task,
            'model_name': self.cfg.model.name,
            'model_id': self.model.id,
            'model_params': str(self.model),
            'tr': {
                'learning': {'mse': None, 'r2': None, 'mae': None, 'mape': None},
                'ranking': [],
            },
            'val': {
                'learning': {'mse': None, 'r2': None, 'mae': None, 'mape': None},
                'ranking': []
            },
            'test': {
                'learning': {'mse': None, 'r2': None, 'mae': None, 'mape': None},
                'ranking': []
            },
            'time': {
                'train': 0.0,
                'test': 0.0,
                'eval': 0.0
            }
        }

    # @staticmethod
    # def _get_results_store():
    #     return {
    #         'tr': {
    #             'loss': []
    #         },
    #         'val': {
    #             'loss': []
    #         },
    #         'test': {
    #             'loss': []
    #         },
    #         'best': {
    #             'loss': {'total': np.infty, 'rank': np.infty, 'weight': np.infty, 'time': np.infty},
    #             'epoch': None
    #         },
    #         'time': {
    #             'train': 0.0,
    #             'test': 0.0,
    #             'eval': 0.0
    #         }
    #     }
