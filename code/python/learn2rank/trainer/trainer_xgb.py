import logging
from pathlib import Path

from sklearn.datasets import load_svmlight_file

from learn2rank.utils.metrics import eval_order_metrics
from learn2rank.utils.metrics import eval_rank_metrics
from learn2rank.utils.order import pred_score2order
from learn2rank.utils.order import pred_score2rank
from .trainer import Trainer

log = logging.getLogger(__name__)
import pandas as pd

class XGBoostTrainer(Trainer):
    def __init__(self, data=None, model=None, cfg=None):
        super().__init__(data, model, cfg)
        log.setLevel(getattr(logging, cfg.logging_level.upper()))
        # Load files
        self.data = Path(data)

        dataset_str = 'dataset_pair_svmrank'
        names_str = 'names_pair_svmrank'
        n_items_str = 'n_items_pair_svmrank'

        if self.cfg.dataset.fused:  # Load fused dataset
            dataset_str += '_all'
            names_str += '_all'
            n_items_str += '_all'

            self.train_data_file = str(self.data / f'{dataset_str}_train.dat')
            self.val_data_file = str(self.data / f'{dataset_str}_val.dat')
            self.test_data_file = self.data / f'{dataset_str}_test.dat'
            self.test_data_file = str(self.test_data_file) if self.test_data_file.exists() else None

            self.train_n_items_file = self.data / f'{n_items_str}_train.dat'
            self.val_n_items_file = self.data / f'{n_items_str}_val.dat'
            self.test_n_items_file = self.data / f'{n_items_str}_test.dat'

            self.train_names_file = self.data / f'{names_str}_train.dat'
            self.val_names_file = self.data / f'{names_str}_val.dat'
            self.test_names_file = self.data / f'{names_str}_test.dat'

        else:  # Load dataset of a given size
            self.train_data_file = str(self.data / f'{self.cfg.problem.size}_{dataset_str}_train.dat')
            self.val_data_file = str(self.data / f'{self.cfg.problem.size}_{dataset_str}_val.dat')
            self.test_data_file = str(self.data / f'{self.cfg.problem.size}_{dataset_str}_test.dat')

            self.train_n_items_file = self.data / f'{self.cfg.problem.size}_{n_items_str}_train.dat'
            self.val_n_items_file = self.data / f'{self.cfg.problem.size}_{n_items_str}_val.dat'
            self.test_n_items_file = self.data / f'{self.cfg.problem.size}_{n_items_str}_test.dat'

            self.train_names_file = self.data / f'{self.cfg.problem.size}_{names_str}_train.dat'
            self.val_names_file = self.data / f'{self.cfg.problem.size}_{names_str}_val.dat'
            self.test_names_file = self.data / f'{self.cfg.problem.size}_{names_str}_test.dat'

        # Process files
        self.x_train, self.y_train = load_svmlight_file(self.train_data_file)
        self.x_val, self.y_val = load_svmlight_file(self.val_data_file)

        # Load test if it exists
        self.x_test = None
        if self.test_data_file is not None:
            self.x_test, _ = load_svmlight_file(self.test_data_file)

        self.ps = self._get_preds_store()
        self.rs = self._get_results_store()
        self.rs['task'] = self.cfg.task
        self.rs['model_name'] = self.cfg.model.name

        self.ps['tr']['names'] = self.train_names_file.read_text().strip().split('\n')
        self.ps['val']['names'] = self.val_names_file.read_text().strip().split('\n')
        if self.test_data_file is not None:
            self.ps['test']['names'] = self.test_names_file.read_text().strip().split('\n')
        self.ps['tr']['n_items'] = list(map(int, self.train_n_items_file.read_text().strip().split('\n')))

        self.ps['val']['n_items'] = list(map(int, self.val_n_items_file.read_text().strip().split('\n')))
        if self.test_data_file is not None:
            self.ps['test']['n_items'] = list(map(int, self.val_n_items_file.read_text().strip().split('\n')))

        unflattend = list(map(self.unflatten_data,
                              (self.x_train, self.y_train, self.x_val, self.y_val),
                              (self.ps["tr"]["n_items"],
                               self.ps["tr"]["n_items"],
                               self.ps["val"]["n_items"],
                               self.ps["val"]["n_items"])))
        self.x_train_uf, self.y_train_uf = unflattend[0], unflattend[1]
        self.x_val_uf, self.y_val_uf = unflattend[2], unflattend[3]
        self.x_test_uf = []
        if self.x_test is not None:
            self.x_test_uf = self.unflatten_data(self.x_test, self.ps["test"]["n_items"])

    def run(self):
        self.model.fit(
            self.x_train,
            self.y_train,
            self.ps["tr"]["n_items"],
            eval_set=[(self.x_val, self.y_val)],
            eval_group=[self.ps["val"]["n_items"]],
        )

        # Train pred
        for x in self.x_train_uf:
            self.ps["tr"]["score"].append(self.model.predict(x))

        # Val pred
        for x in self.x_val_uf:
            self.ps["val"]["score"].append(self.model.predict(x))

        for x in self.x_test_uf:
            self.ps["test"]["score"].append(self.model.predict(x))

        # Eval learning metrics
        # log.info(f"* {self.cfg.model.name} Results")
        # log.info("** Train learning metrics:")
        # self.rs["tr"]["learning"] = self.eval_learning_metrics(split="train")
        # log.info("** Validation learning metrics:")
        # self.rs["val"]["learning"] = self.eval_learning_metrics(split="val")

        # Transform scores to order and ranks
        train_order = pred_score2order(self.y_train_uf, reverse=True)
        self.ps["tr"]["order"] = pred_score2order(self.ps["tr"]["score"], reverse=True)
        train_rank = pred_score2rank(self.y_train_uf, reverse=True)
        self.ps["tr"]["rank"] = pred_score2rank(self.ps["tr"]["score"], reverse=True)

        val_order = pred_score2order(self.y_val_uf, reverse=True)
        self.ps["val"]["order"] = pred_score2order(self.ps["val"]["score"], reverse=True)
        val_rank = pred_score2rank(self.y_val_uf, reverse=True)
        self.ps["val"]["rank"] = pred_score2rank(self.ps["val"]["score"], reverse=True)

        if self.x_test is not None:
            self.ps["test"]["order"] = pred_score2order(self.ps["test"]["score"], reverse=True)
            self.ps["test"]["rank"] = pred_score2rank(self.ps["test"]["score"], reverse=True)

        # Eval rank predictions
        log.info("** Train order metrics:")

        self.rs["tr"]["ranking"].extend(eval_order_metrics(train_order,
                                                           self.ps["tr"]["order"],
                                                           self.ps["tr"]["n_items"]))
        self.rs["tr"]["ranking"].extend(eval_rank_metrics(train_rank,
                                                          self.ps["tr"]["rank"],
                                                          self.ps["tr"]["n_items"]))

        log.info("** Val order metrics:")
        self.rs["val"]["ranking"].extend(eval_order_metrics(val_order,
                                                            self.ps["val"]["order"],
                                                            self.ps["val"]["n_items"]))
        self.rs["val"]["ranking"].extend(eval_rank_metrics(val_rank,
                                                           self.ps["val"]["rank"],
                                                           self.ps["val"]["n_items"]))


        log.info(f"  {self.cfg.model.name} train time: {self.rs['time']['train']} \n")

        if self.cfg.save:
            self._save_model()
            self._save_predictions()
            self._save_results()

        df = pd.DataFrame(self.rs["val"]["ranking"], columns=['id', 'metric_type', 'metric_value'])
        return df[df['metric_type'] == 'kendall-coeff']['metric_value'].mean()

    def predict(self, *args, **kwargs):
        pass

    @staticmethod
    def unflatten_data(x, group):
        x_unflat = []

        i = 0
        for g in group:
            x_unflat.append(x[i: i + g])
            i += g

        return x_unflat

    def eval_learning_metrics(self, split="train"):
        pass

    @staticmethod
    def _get_preds_store():
        return {
            "tr": {"names": [], "n_items": [], "score": [], "rank": [], "order": []},
            "val": {"names": [], "n_items": [], "score": [], "rank": [], "order": []},
            "test": {"names": [], "n_items": [], "score": [], "rank": [], "order": []}
        }

    @staticmethod
    def _get_results_store():
        return {
            "tr": {
                "learning": {"mse": None, "r2": None, "mae": None, "mape": None},
                "ranking": [],
            },
            "val": {
                "learning": {"mse": None, "r2": None, "mae": None, "mape": None},
                "ranking": [],
            },
            "test": {
                "learning": {"mse": None, "r2": None, "mae": None, "mape": None},
                "ranking": [],
            },
            "time": {"train": 0.0, "test": 0.0, "eval": 0.0},
        }

    def _save_model(self):
        if self.cfg.dataset.fused:
            model_path = self.res_path / f'pretrained/{self.cfg.problem.name}'
        else:
            model_path = self.res_path / f'pretrained/{self.cfg.problem.name}/{self.cfg.problem.size}'
        model_path.mkdir(parents=True, exist_ok=True)
        model_path = model_path / f'model_{self.model.id}.txt'

        self.model.save_model(model_path)
