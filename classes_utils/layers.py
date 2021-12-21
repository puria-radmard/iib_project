import torch
from torch import nn
from util_functions.prob import reparameterised_draw
device = (
    torch.device('cuda') if torch.cuda.is_available() 
    else torch.device('cpu')
)

__all__ = [
    "BidirectionalLSTMHiddenStateStacker",
    "ReparameterisationLayer",
    "EmptyLayer",
    "AdditiveGaussianLayer",
    "MultiplicativeGaussianLayer",
]


class BidirectionalLSTMHiddenStateStacker(nn.Module):

    # Example here is:
    #_, ((h_forward, h_backward), _) = self.encoder(x)
    #h_full = torch.hstack([h_forward, h_backward])

    def __init__(self, cell_type):
        super(BidirectionalLSTMHiddenStateStacker, self).__init__()
        self.cell_type=cell_type

    def forward(self, x):
        if self.cell_type == 'lstm':
            _, (h, _) = x
            batch_size = h.shape[1]
            return torch.hstack([lh for lh in h])
        elif self.cell_type == 'gru':
            _, h = x
            batch_size = h.shape[1]
            return torch.hstack([lh for lh in h])


class BidirectionalLSTMOutputSelector(nn.Module):

    def __init__(self, cell_type):
        super(BidirectionalLSTMOutputSelector, self).__init__()
        self.cell_type=cell_type

    def forward(self, x):
        output, *_ = x
        return output


class ReparameterisationLayer(nn.Module):

    def __init__(self, z_dim):
        raise Exception('Need to generalise to all shapes')
        self.z_dim = z_dim
        super(ReparameterisationLayer, self).__init__()

    def forward(self, x):
        mean = x[:, : self.z_dim]
        if self.training:
            log_var = x[:, self.z_dim :]
            z = reparameterised_draw(mean, log_var)
            return z
        else:
            return mean


class EmptyLayer(nn.Module):

    def __init__(self, *args, **kwargs):
        super(EmptyLayer, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x


class AdditiveGaussianLayer(nn.Module):
    def __init__(self, a):
        super(AdditiveGaussianLayer, self).__init__()
        self.a = a

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        return x + self.a * torch.randn(*x.size(), device = device)


class MultiplicativeGaussianLayer(nn.Module):
    def __init__(self, a):
        super(MultiplicativeGaussianLayer, self).__init__()
        self.a = a

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        return x * (self.a * torch.randn(*x.size(), device = device) + 1)