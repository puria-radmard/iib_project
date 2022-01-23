import torch
from torch import nn
from classes_utils.layers import ReparameterisationLayer, EmptyLayer

__all__ = [
    'DecoderBase',
    'AudioEncoderBase',
]

class DecoderBase(nn.Module):
    def __init__(self, mean_first):
        super(DecoderBase, self).__init__()
        self.mean_first = mean_first

    def forward_method(self, x, *args, **kwargs):
        raise NotImplementedError

    def forward(self, embeddings, *args, **kwargs):
        if self.mean_first:
            embeddings = [torch.mean(torch.stack(embeddings), dim=0)]
        decodings = [self.forward_method(e, *args, **kwargs) for e in embeddings]
        return decodings


class SkipDecoderBase(DecoderBase):

    def forward(self, embeddings, skip_list, *args, **kwargs):
        if self.mean_first:
            raise Exception('Cannot mean decodings if using skips')
        decodings = [self.forward_method(e, sl, *args, **kwargs) for e, sl in zip(embeddings, skip_list)]
        return decodings


class EncoderBase(nn.Module):
    def __init__(self, layers, dropout_idxs, noise_idx, variational, embedding_dim, return_skips=False):
        super(EncoderBase, self).__init__()
        self.layers = layers
        self.dropout_idxs = dropout_idxs
        self.noise_idx = noise_idx
        self.variational = variational
        self.reparameterisation_layer = ReparameterisationLayer(embedding_dim//2) if variational else None
        self.return_skips = return_skips
        self.embedding_dim = embedding_dim

    def eval(self, keep_dropouts_on):
        super().eval()
        if keep_dropouts_on:
            for i, l in enumerate(self.layers):
                if i in self.dropout_idxs:
                    l.train()
        return self

    def forward(self, x):
        
        x = x.float()

        skip_list = [x]
        for l in self.layers:
            x = l(x)
            # If we want to communicate layer outputs to the decoder, save each one
            if self.return_skips:
                skip_list.append(x)

        # Various return options so this might make it easier
        return_list = [x]

        # If this is part of a VAE, sample z from x and return both x and z
        if self.variational:
            z = self.reparameterisation_layer(x)
            return_list.append(z)
        
        if self.return_skips:
            return_list.append(skip_list)

        # Used to x coming out alone if no skips and not variational
        return tuple(return_list) if len(return_list) > 1 else x


class ImageEncoderBase(EncoderBase):

    def __init__(self, input_size, layers, dropout_idxs, noise_idx, variational, embedding_dim, return_skips):
        self.input_size=input_size
        super(ImageEncoderBase, self).__init__(
            layers, dropout_idxs, noise_idx, variational, embedding_dim, return_skips
        )


class AudioEncoderBase(EncoderBase):

    def __init__(self, mfcc_dim, layers, dropout_idxs, noise_idx, variational, embedding_dim):
        self.mfcc_dim = mfcc_dim
        super(AudioEncoderBase, self).__init__(
            layers, dropout_idxs, noise_idx, variational, embedding_dim
        )

    def forward(self, x):
        x = x.float()
        # x = x.permute(0, 2, 1)
        return super().forward(x)
