import torch

from .loss import Loss


class RMSE(Loss):
    def compute(self, y_pred, y):
        y_pred = y_pred.clone()
        y_true = y.clone()

        errors = (y_true - y_pred)
        squared_errors = errors ** 2
        mean_squared_errors = torch.mean(squared_errors, dim=1)
        rmses = torch.sqrt(mean_squared_errors)

        return torch.mean(rmses)
