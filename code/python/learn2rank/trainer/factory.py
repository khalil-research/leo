from learn2rank.utils import Factory
from .trainer_minwt import MinWeightTrainer
from .trainer_pytorch import PyTorchTrainer
from .trainer_sklearn import SklearnTrainer
from .trainer_svmrank import SVMRankTrainer

# Factory to hold trainers
trainer_factory = Factory()
trainer_factory.register_member('PyTorchTrainer', PyTorchTrainer)
trainer_factory.register_member('SklearnTrainer', SklearnTrainer)
trainer_factory.register_member('SVMRankTrainer', SVMRankTrainer)
trainer_factory.register_member('MinWeightTrainer', MinWeightTrainer)
