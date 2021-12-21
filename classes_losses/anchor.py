import torch
from torch import nn

__all__ = [
    'EncoderEnsembleAnchorLoss'
]

class EncoderEnsembleAnchorLoss(nn.Module):
    def __init__(self, sum = True):
        super(EncoderEnsembleAnchorLoss, self).__init__()
        reduction = "sum" if sum else "none"
        self.MSE = nn.MSELoss(reduction=reduction)

    def forward(self, embedding_samples, sum_losses = True):
        mean = torch.mean(torch.stack(embedding_samples), dim=0)
        losses = [self.MSE(mean, e) for e in embedding_samples]
        return sum(losses) if sum_losses else losses
