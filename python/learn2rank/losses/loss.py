from abc import ABC, abstractmethod

import torch


class Loss(ABC):
    padded_value_indicator = -1
    eps = 1e-10

    @abstractmethod
    def compute(self, y_pred, y):
        """
        Compute the loss value based on prediction and ground truth values
        Parameters
        ----------
        y_pred: torch.Tensor of shape batch size x max length
            Prediction
        y: torch.Tensor of shape batch size x max length
            Ground truth
        Returns
        -------
        loss: torch.Tensor
        """
        pass
