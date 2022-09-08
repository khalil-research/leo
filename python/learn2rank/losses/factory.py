from learn2rank.utils import Factory
from .listnet import ListNetLoss
from .pointwise import PointwiseRMSELoss
from .ranknet import RankNetLoss
from .rmse import RMSE

loss_factory = Factory()
loss_factory.register_member('RMSE', RMSE)
loss_factory.register_member('PointwiseRMSELoss', PointwiseRMSELoss)
loss_factory.register_member('RankNetLoss', RankNetLoss)
loss_factory.register_member('ListNetLoss', ListNetLoss)
