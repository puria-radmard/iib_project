from torch import nn
import torch
from util_functions.prob import gaussian_KL

__all__ = [
    'VAEEnsemblePriorLoss'
]

class VAEEnsemblePriorLoss(nn.Module):
    def __init__(self, hidden_size):
        super(VAEEnsemblePriorLoss, self).__init__()
        self.z_dim = int(hidden_size / 2)
        self.cov = torch.eye(self.z_dim)
        self.mean = torch.zeros(self.z_dim)

    def forward(self, encodings, zs, batch, **kwargs):
        raise Exception
        # zs come in a list of tensors size [batch_size, z_dim*2], one for each ensemble member
        losses = []
        # iterate through list and find KL(z||unit)
        means = [z[:,:self.z_dim] for z in encodings]
        vars_ = [torch.exp(0.5*z[:,self.z_dim:]) for z in encodings]
        logd_terms = [torch.log(1/torch.prod(v, -1)) for v in vars_]
        trce_terms = [torch.sum(v, -1) for v in vars_]
        quad_terms = [(m*m).sum(-1) for m in means]
        losses = [
            0.5*(logd_terms[j]-self.z_dim+trce_terms[j]+quad_terms[j])
            for j in range(len(encodings))
        ]
        return torch.sum(torch.stack(losses))
