from itertools import product

import torch
from torch.nn import BCEWithLogitsLoss

from .loss import Loss


class RankNetLoss(Loss):
    def __init__(self, weight='None'):
        super(RankNetLoss, self).__init__()
        self.weight = weight

    def compute(self, y_pred, y):
        """
        RankNet loss introduced in "Learning to Rank using Gradient Descent".
        """
        y_pred = y_pred.clone()
        y_true = y.clone()

        mask = y_true == self.padded_value_indicator
        y_pred[mask] = float('-inf')
        y_true[mask] = float('-inf')

        # here we generate every pair of indices from the range of document length in the batch
        document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

        pairs_true = y_true[:, document_pairs_candidates]
        selected_pred = y_pred[:, document_pairs_candidates]

        # here we calculate the relative true relevance of every candidate pair
        true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
        pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

        # here we filter just the pairs that are 'positive' and did not involve a padded instance
        # we can do that since in the candidate pairs we had symetric pairs so we can stick with
        # positive ones for a simpler loss function formulation
        the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

        pred_diffs = pred_diffs[the_mask]

        if self.weight == 'diff':
            abs_diff = torch.abs(true_diffs)
            weight = abs_diff[the_mask]
        elif self.weight == 'diff_power':
            true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
            abs_diff = torch.abs(true_pow_diffs)
            weight = abs_diff[the_mask]
        else:
            weight = None

        # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
        # whether one document is better than the other and not about the actual difference in
        # their relevancy levels
        true_diffs = (true_diffs > 0).type(torch.float32)
        true_diffs = true_diffs[the_mask]

        return BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)
