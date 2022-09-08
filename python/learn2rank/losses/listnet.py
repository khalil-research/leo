import torch
import torch.nn.functional as F

from .loss import Loss


class ListNetLoss(Loss):
    def compute(self, y_pred, y_true):
        """
        ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
        """
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        mask = y_true == self.padded_value_indicator
        y_pred[mask] = float('-inf')
        y_true[mask] = float('-inf')

        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        preds_smax = preds_smax + self.eps
        preds_log = torch.log(preds_smax)

        return torch.mean(-torch.sum(true_smax * preds_log, dim=1))
