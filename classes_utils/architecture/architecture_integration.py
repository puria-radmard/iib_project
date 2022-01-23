import torch
from torch import nn
from util_functions.base import load_state_dict
from config.ensemble import ensemble_method_dict
from config import architectures

__all__ = [
    'EncoderDecoderEnsemble',
    'AudioEncoderDecoderEnsemble',
    'SkipEncoderDecoderEnsemble'
]

device = (
    torch.device('cuda') if torch.cuda.is_available() 
    else torch.device('cpu')
)

class EncoderDecoderEnsemble(nn.Module):
    def __init__(self, ensemble_type, encoder_type, decoder_type, ensemble_size,
                encoder_ensemble_kwargs, decoder_ensemble_kwargs, mult_noise=0, weights_path=None):
        super(EncoderDecoderEnsemble, self).__init__()
        
        # Get type of ensemble, e.g. simple, dropout, noise, etc
        ensemble_class = ensemble_method_dict[ensemble_type]

        # Get encoder and decoder types
        encoder_class = getattr(architectures, encoder_type)
        decoder_class = getattr(architectures, decoder_type)

        # Build encoder ensemble object
        self.encoder_ensemble = ensemble_class(
            encoder_type = encoder_class, ensemble_size = ensemble_size, 
            encoder_ensemble_kwargs = encoder_ensemble_kwargs
        )

        # Build single decoder object
        self.decoder = decoder_class(**decoder_ensemble_kwargs)

        # Add attributes for method calls later
        self.variational = self.encoder_ensemble.variational
        self.mult_noise = mult_noise

        load_state_dict(self, weights_path)

    def eval(self):
        super().eval()
        self.encoder_ensemble.eval()

    def add_noise(self, x):
        # Only want to add noise in 
        if self.mult_noise > 0:
            noise = self.mult_noise * torch.randn(x.shape, device=device)
            x_in = x + noise
        else:
            x_in = x
        return x_in

    def forward_method(self, x_in):
        if self.variational:
            # encoder returns encoded_list, zs, skips
            encodings, zs = self.encoder_ensemble(x_in)
            decodings = self.decoder(zs)
            return encodings, zs, decodings
        else:
            # encoder returns encoded_list, skips
            encodings = self.encoder_ensemble(x_in)
            decodings = self.decoder(encodings)
            return encodings, decodings

    def forward(self, x):
        x_in = self.add_noise(x)
        return self.forward_method(x_in)


class AudioEncoderDecoderEnsemble(EncoderDecoderEnsemble):

    def forward_method(self, x_in):
        _, seq_len, _ = x_in.shape
        # All of these are lists
        if self.variational:
            encodings, zs = self.encoder_ensemble(x_in)
            decodings = self.decoder(zs, seq_len)
            return encodings, zs, decodings
        else:
            encodings = self.encoder_ensemble(x_in)
            decodings = self.decoder(encodings, seq_len)
            return encodings, decodings


class SkipEncoderDecoderEnsemble(EncoderDecoderEnsemble):

    def forward_method(self, x_in):
        if self.variational:
            # encoder returns encoded_list, zs, skips
            encodings, zs, skip_list = self.encoder_ensemble(x_in)
            decodings = self.decoder(zs, skip_list)
            return encodings, zs, decodings
        else:
            # encoder returns encoded_list, skips
            encodings, skip_list = self.encoder_ensemble(x_in)
            decodings = self.decoder(encodings, skip_list)
            return encodings, decodings
