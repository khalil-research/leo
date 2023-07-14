from leo.utils import Factory
from .trainer_heuristic_order import HeuristicOrderTrainer
from .trainer_lex import LexTrainer
from .trainer_random import RandomTrainer
from .trainer_sklearn import SklearnTrainer
from .trainer_smac_dataset import SmacDTrainer
from .trainer_smac_inst import SmacITrainer
from .trainer_svmrank import SVMRankTrainer
from .trainer_xgb import XGBoostTrainer

# Factory to hold trainers
trainer_factory = Factory()
trainer_factory.register_member('SklearnTrainer', SklearnTrainer)
trainer_factory.register_member('XGBoostTrainer', XGBoostTrainer)
trainer_factory.register_member('SVMRankTrainer', SVMRankTrainer)
trainer_factory.register_member('SmacITrainer', SmacITrainer)
trainer_factory.register_member('SmacDTrainer', SmacDTrainer)
trainer_factory.register_member('LexTrainer', LexTrainer)
trainer_factory.register_member('HeuristicOrderTrainer', HeuristicOrderTrainer)
trainer_factory.register_member('RandomTrainer', RandomTrainer)
