import torch

from .loss import Loss


class PointwiseRMSELoss(Loss):
    def compute(self, y_pred, y):
        y_pred = y_pred.clone()
        y_true = y.clone()

        mask = y_true == self.padded_value_indicator
        valid_mask = (y_true != self.padded_value_indicator).type(torch.float32)
        # no_of_levels = valid_mask - 1

        y_true[mask] = 0
        y_pred[mask] = 0

        errors = (y_true - y_pred)

        squared_errors = errors ** 2

        mean_squared_errors = torch.sum(squared_errors, dim=1) / torch.sum(valid_mask, dim=1)

        rmses = torch.sqrt(mean_squared_errors)

        return torch.mean(rmses)
