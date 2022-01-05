import torch
from torch import nn

# https://arxiv.org/pdf/1508.01211.pdf

class pBLSTMLayer(nn.Module):

    """
        Instead of a typical BLSTM: h^j_i = BLSTM(h^j_{i-1}, h_i^{j−1})
        We have: h^j_i = pBLSTM (h^j_{i-1}, [h^{j−1}_{2i}, h^{j−1}_{2i+1}])

            for time step i in layer j

        i.e. instead of feeding previous layer directly, feed in concatentation of consecutive layers
    """

    def __init__(
        self, input_size, hidden_size, num_layers, dropout, 
    ):
        super(pBLSTMLayer, self).__init__()

        self.BLSTM = nn.LSTM(
            input_size=input_size * 2, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        batch, dur, feat = x.shape
        time_reduc = int(dur / 2)
        try:
            input_xr = x.contiguous().view(batch, time_reduc, feat * 2)
        except:
            import pdb; pdb.set_trace()
        output, hidden = self.BLSTM(input_xr)
        return output# , hidden


class pLSTMOutputExtractor(nn.Module):

    def __init__(self, _print=False):
        super(pLSTMOutputExtractor, self).__init__()
        self.print = _print

    def forward(self, p_lstm_output):
        output, hidden = p_lstm_output
        if self.print:
            print(output.shape, hidden[0].shape, hidden[1].shape)
        return output


class PrepForPyramid(nn.Module):

    def __init__(self, pyramid_size:int, layer_reduction:int=2):
        self.factor = layer_reduction**pyramid_size
        super(PrepForPyramid, self).__init__()

    def forward(self, x):
        B, L, *pad_shape = x.shape
        frames_to_add = self.factor - (L % self.factor)
        padding = torch.zeros(B, frames_to_add, *pad_shape, dtype=x.dtype)
        return torch.hstack([x, padding.to(x.device)])
