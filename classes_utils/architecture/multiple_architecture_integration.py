import torch
from torch import nn
from util_functions.base import load_state_dict
from config.ensemble import ensemble_method_dict
from config.architectures import encoder_types, decoder_types


__all__ = [
    'MultipleEncoderDecoderEnsemble'
]

device = (
    torch.device('cuda') if torch.cuda.is_available() 
    else torch.device('cpu')
)


class MultipleEncoderDecoderEnsemble(nn.Module):
    def __init__(self, ensemble_type, encoder_type, decoder_names, ensemble_size,
                encoder_ensemble_kwargs, decoder_ensemble_kwargs_list, mult_noise=0, weights_path=None):

        super(MultipleEncoderDecoderEnsemble, self).__init__()
        
        ensemble_class = ensemble_method_dict[ensemble_type]
        encoder_class = encoder_types[encoder_type]
        self.encoder_ensemble = ensemble_class(
            encoder_type = encoder_class, ensemble_size = ensemble_size, 
            encoder_ensemble_kwargs = encoder_ensemble_kwargs
        )

        decoders = []
        decoder_iterator = zip(decoder_names, decoder_ensemble_kwargs_list)
        for decoder_name, decoder_ensemble_kwargs in decoder_iterator:
            decoder_class = decoder_types[decoder_name]
            decoder = decoder_class(**decoder_ensemble_kwargs)   
            decoders.append(decoder)
        
        self.decoders = nn.ModuleList(decoders)

        self.variational = self.encoder_ensemble.variational
        self.mult_noises = mult_noise

        # Which encoder/decoders are going to be used
        self.decoder_flags = []

        load_state_dict(self, weights_path)

    def eval(self, include_encoder, decoder_indices):
        if include_encoder:
            self.encoder_ensemble.eval()
        for j in decoder_indices:
            self.decoders[j].eval()

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
            decodings = [self.decoders[i](zs) for i in self.decoder_flags]
            decodings = decodings[0] if len(self.decoder_flags) == 1 else decodings
            return encodings, zs, decodings
        else:
            # encoder returns encoded_list, skips
            encodings = self.encoder_ensemble(x_in)
            decodings = self.decoder(encodings)
            decodings = [self.decoders[i](encodings) for i in self.decoder_flags]
            decodings = decodings[0] if len(self.decoder_flags) == 1 else decodings
            return encodings, decodings

    def pick_decoders(self, *dis):
        self.decoder_flags = dis

    def forward(self, x):
        x_in = self.add_noise(x)
        return self.forward_method(x_in)
