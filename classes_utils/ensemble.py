import torch
from torch import nn
from classes_utils.layers import ReparameterisationLayer, MultiplicativeGaussianLayer, AdditiveGaussianLayer
from util_functions.base import load_state_dict

__all__ = [
    'AudioEncoderEnsembleBase',
    'AudioEncoderBasicEnsemble',
    'AudioEncoderDropoutEnsemble',
    'AudioEncoderMultiplicativeNoiseEnsemble',
    'AudioEncoderAdditiveNoiseEnsemble',
]

class AudioEncoderEnsembleBase(nn.Module):
    def __init__(self, encoder_ensemble, ensemble_size, **kwargs):
        super(AudioEncoderEnsembleBase, self).__init__()
        self.encoder_ensemble = encoder_ensemble
        self.ensemble_size = ensemble_size
        self.return_skips = getattr(self.get_encoder(), 'return_skips', False)
        self.__dict__.update(kwargs)

    def eval(self):
        super().eval()
        if isinstance(self.encoder_ensemble, nn.ModuleList):
            [e.eval(keep_dropouts_on = self.keep_dropouts_on) for e in self.encoder_ensemble]
        else:
            self.encoder_ensemble.eval(keep_dropouts_on = self.keep_dropouts_on)

    def get_encoder(self):        
        # Get one of the encoders in the ensemble. For almost all cases it's this simple
        return self.encoder_ensemble

    def forward_method(self, x):
        return NotImplementedError

    def forward(self, x):
        # This is a list of self.ensemble_size tuples
        # Each tuple t has t[0] = encoding
        # Then, it has z and/or skip_list as determined by encoder mode
        output_list = self.forward_method(x)
        if self.variational or self.return_skips:
            return list(zip(*output_list))
        else:
            return output_list


class AudioEncoderBasicEnsemble(AudioEncoderEnsembleBase):
    def __init__(self, encoder_type, ensemble_size, encoder_ensemble_kwargs, weights_path = None):
        encoder_ensemble = nn.ModuleList([
            encoder_type(**encoder_ensemble_kwargs) for _ in range(ensemble_size)
        ])
        super(AudioEncoderBasicEnsemble, self).__init__(
            encoder_ensemble, ensemble_size, **encoder_ensemble_kwargs, keep_dropouts_on = False
        )
        load_state_dict(self, weights_path)

    def get_encoder(self):        
        # Get one of the encoders in the ensemble. Only unusual case
        return self.encoder_ensemble[0]

    def forward_method(self, x):
        encoded = [member(x) for member in self.encoder_ensemble]
        return encoded


class AudioEncoderDropoutEnsemble(AudioEncoderEnsembleBase):
    def __init__(self, encoder_type, ensemble_size, encoder_ensemble_kwargs, weights_path = None):
        encoder_ensemble = encoder_type(**encoder_ensemble_kwargs)
        super(AudioEncoderDropoutEnsemble, self).__init__(
            encoder_ensemble, ensemble_size, **encoder_ensemble_kwargs, keep_dropouts_on = True
        )
        load_state_dict(self, weights_path)

    def forward_method(self, x):
        encoded = [self.encoder_ensemble(x) for _ in range(self.ensemble_size)]
        return encoded


class AudioEncoderMultiplicativeNoiseEnsemble(AudioEncoderEnsembleBase):
    def __init__(self, encoder_type, ensemble_size, encoder_ensemble_kwargs, weights_path = None):
        encoder_ensemble = encoder_type(**encoder_ensemble_kwargs)
        super(AudioEncoderMultiplicativeNoiseEnsemble, self).__init__(
            encoder_ensemble, ensemble_size, **encoder_ensemble_kwargs, keep_dropouts_on = True
        )
        self.noise_layer = MultiplicativeGaussianLayer(self.a)
        load_state_dict(self, weights_path)
        raise Exception('Need to renovate this for sliding window models')

    def pre_noise_forward(self, x):
        for i in range(self.encoder_ensemble.noise_idx + 1):
            x = self.encoder_ensemble.layers[i](x)
        return x

    def post_noise_forward(self, x):
        x = self.noise_layer(x)
        for i in range(self.encoder_ensemble.noise_idx + 1, len(self.encoder_ensemble.layers)):
            x = self.encoder_ensemble.layers[i](x)
        return x

    def forward_method(self, x):
        pre_noise_encoded = self.pre_noise_forward(x)
        encoded = [self.post_noise_forward(pre_noise_encoded) for _ in range(self.ensemble_size)]
        return encoded


class AudioEncoderAdditiveNoiseEnsemble(AudioEncoderEnsembleBase):
    def __init__(self, encoder_type, ensemble_size, encoder_ensemble_kwargs, weights_path = None):
        encoder_ensemble = encoder_type(**encoder_ensemble_kwargs)
        super(AudioEncoderAdditiveNoiseEnsemble, self).__init__(
            encoder_ensemble, ensemble_size, **encoder_ensemble_kwargs, keep_dropouts_on = True
        )
        self.noise_layer = AdditiveGaussianLayer(self.a)
        load_state_dict(self, weights_path)
        raise Exception('Need to renovate this for sliding window models')

    def pre_noise_forward(self, x):
        for i in range(self.encoder_ensemble.noise_idx):
            x = self.encoder_ensemble.layers[i](x)
        return x

    def post_noise_forward(self, x):
        x = self.noise_layer(x)
        for i in range(self.encoder_ensemble.noise_idx + 1, len(self.encoder_ensemble.layers)):
            x = self.encoder_ensemble.layers[i](x)
        return x

    def forward_method(self, x):
        pre_noise_encoded = self.pre_noise_forward(x)
        encoded = [self.post_noise_forward(pre_noise_encoded) for _ in range(self.ensemble_size)]
        return encoded
