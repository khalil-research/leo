import torch


class WeightPenalty:
    """Apply weight penalty (L1/L2/Both) to model parameters"""

    def __init__(self, weights=None, include_bias=False):
        self.weights = {'lasso': 1e-3, 'ridge': 1e-3} \
            if weights is None else weights
        self.include_bias = include_bias

    def compute(self, model):
        penalty = self.elastic_penalty(model)
        return penalty

    def lasso_penalty(self, model):
        """L1 penalty"""
        penalty = 0
        for name, param in model.named_parameters():
            if 'weight' in name or self.include_bias:
                penalty += torch.norm(param, 1)

        return self.weights['lasso'] * penalty

    def ridge_penalty(self, model):
        """L2 penalty"""
        penalty = 0
        for name, param in model.named_parameters():
            if 'weight' in name or self.include_bias:
                penalty += torch.pow(torch.norm(param, 2), 2)

        return self.weights['ridge'] * penalty

    def elastic_penalty(self, model):
        """Weighted combination of L1 and L2 penalty"""
        l1 = self.lasso_penalty(model)
        l2 = self.ridge_penalty(model)
        return l1 + l2
