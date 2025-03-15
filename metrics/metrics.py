from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric
from typing import Dict

import torch

class RMSLE(MultiHorizonMetric):
    """
    Root mean square error

    Defined as ``(y_pred - target)**2``
    """

    def __init__(self, reduction="sqrt-mean", **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def loss(self, y_pred: Dict[str, torch.Tensor], target):
        point_pred = self.to_prediction(y_pred)
        zeroes = torch.zeros_like(point_pred)
        
        point_pred_with_cutoff = torch.max(point_pred, zeroes)
        loss = torch.log1p(point_pred_with_cutoff) - torch.log1p(target)
        return loss