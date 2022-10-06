from learn2rank.utils import Factory
from .trainer_pytorch import PyTorchTrainer
from .trainer_sklearn import SklearnTrainer

# Factory to hold trainers
trainer_factory = Factory()
trainer_factory.register_member('PyTorchTrainer', PyTorchTrainer)
trainer_factory.register_member('SklearnTrainer', SklearnTrainer)
