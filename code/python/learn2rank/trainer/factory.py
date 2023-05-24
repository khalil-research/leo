from learn2rank.utils import Factory
from .trainer_minwt import MinWeightTrainer
from .trainer_pytorch import PyTorchTrainer
from .trainer_sklearn import SklearnTrainer
from .trainer_svmrank import SVMRankTrainer
from .trainer_xgb import XGBoostTrainer
from .trainer_smac import SmacOneTrainer
from .trainer_smac_all import SmacAllTrainer
from .trainer_canonical import CanonicalTrainer

# Factory to hold trainers
trainer_factory = Factory()
trainer_factory.register_member('PyTorchTrainer', PyTorchTrainer)
trainer_factory.register_member('SklearnTrainer', SklearnTrainer)
trainer_factory.register_member('XGBoostTrainer', XGBoostTrainer)
trainer_factory.register_member('SVMRankTrainer', SVMRankTrainer)
trainer_factory.register_member('MinWeightTrainer', MinWeightTrainer)
trainer_factory.register_member('SmacOneTrainer', SmacOneTrainer)
trainer_factory.register_member('SmacAllTrainer', SmacAllTrainer)
trainer_factory.register_member('CanonicalTrainer', CanonicalTrainer)
