import torch, sys
from torch import nn
from torch.nn.modules.conv import Conv2d
from classes_architectures.base import DecoderBase, SkipDecoderBase
from classes_architectures.cifar.encoder import ConvBlock

__all__ = ['UNetDecoder', 'NoSkipDecoder']

DEFAULT_UNET_DECODER_OUT_CHANNELS = [256, 128, 64, 32]
DEFAULT_UNET_DECODER_KERNEL_SIZES = [3, 3, 3, 3, 3]
DEFAULT_UNET_DECODER_STRIDES = [2, 2, 2, 2, 1]
DEFAULT_UNET_DECODER_CONCATS = [-1, 3, 2, 1]

DEFAULT_NOSKIP_FC_SIZES = [2048]
DEFAULT_NOSKIP_RESHAPE_SIZE = (128, 4, 4)
DEFAULT_NOSKIP_DECODER_OUT_CHANNELS = [32, 16]
DEFAULT_NOSKIP_DECODER_KERNEL_SIZES = [4, 4, 2]
DEFAULT_NOSKIP_DECODER_STRIDES = [1, 2, 2]



class DeconvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, concat_idx, padding=1):
        super(DeconvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.concat_idx = concat_idx

    def forward(self, x, output_size=None):
        x = self.conv(x) if output_size is None else self.conv(x, output_size=output_size)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UNetDecoder(SkipDecoderBase):

    def __init__(self, embedding_dim, out_channels, kernel_sizes, strides, concat_idxs, output_channels, flattened = False):
        super(UNetDecoder, self).__init__(mean_first=False)

        # Sanity check on inputs given
        assert (len(out_channels)+1) == len(kernel_sizes) == len(strides) == (len(concat_idxs)+1)

        # This is really an embedding_depth, as you can have 3D embeddings
        self.embedding_dim = embedding_dim

        # First deconv from the embedding dimension to the first output channel
        deconv_layers = [DeconvBlock(embedding_dim, out_channels[0], kernel_sizes[0], strides[0], concat_idxs[0])]

        # Cascade the deconv connections
        for i in range(1, len(out_channels)):
            # Times two if we want skip connections
            in_channels = out_channels[i-1]*2 if concat_idxs[i] >= 0 else out_channels[i-1]
            deconv_layers.append(DeconvBlock(in_channels, out_channels[i], kernel_sizes[i], strides[i], concat_idxs[i]))

        self.deconv_layers = nn.ModuleList(deconv_layers)

        # No ReLU on final deconv, so define them seperately
        self.final_deconv = nn.ConvTranspose2d(
            in_channels=out_channels[-1], out_channels=output_channels, 
            kernel_size=kernel_sizes[-1], stride=strides[-1], padding=1
        )
        self.final_activation = nn.Identity()

        # Flattened is whever we were anticipating a 1D embedding from encoder
        self.flattened = flattened

    def forward_method(self, x, skip_list):

        # Turn into 3D image if embedding flattened
        if self.flattened:
            x = x.view(x.shape[0], self.embedding_dim, 1, 1)
        
        target_size_idx = -1

        # Cycle upsample layers
        for deconv_layer in self.deconv_layers:

            target_size_idx -= 1

            # If we want to collect a skip, concat it here
            if deconv_layer.concat_idx >= 0:
                x = torch.hstack([x, skip_list[deconv_layer.concat_idx]])

            # Easy for UNet: the output size is always the next passed message up
            x = deconv_layer(x, output_size=skip_list[target_size_idx].shape)

        x = self.final_deconv(x)
        x = self.final_activation(x)
        return x


class NoSkipDecoder(DecoderBase):

    def __init__(
        self, fc_sizes, out_channels, kernel_sizes, strides, output_channels, base_padding, reshape_size = None, first_conv_size = None
    ):
        super(NoSkipDecoder, self).__init__(mean_first=False)

        if (reshape_size == first_conv_size == None) or (reshape_size != None and first_conv_size != None):
            raise Exception('Need either reshape_size OR first_conv_size')
        first_conv_size = first_conv_size if first_conv_size is not None else reshape_size[0]

        self.reshape_size = reshape_size
        
        # Fully connected layers
        fc_layers = []
        for i, fcs in enumerate(fc_sizes[1:]):
            fc_layers.append(nn.Linear(fc_sizes[i], fcs))
            fc_layers.append(nn.Tanh())
        self.fc_layers = nn.Sequential(*fc_layers)

        # Deconv layers
        deconv_layers = [DeconvBlock(first_conv_size, out_channels[0], kernel_sizes[0], strides[0], -1, base_padding)]
        for i in range(1, len(out_channels)):
            deconv_layers.append(
                DeconvBlock(out_channels[i-1], out_channels[i], kernel_sizes[i], strides[i], -1, base_padding)
            )
        self.deconv_layers = nn.Sequential(*deconv_layers)

        # No ReLU on final deconv, so define them seperately
        self.final_deconv = nn.ConvTranspose2d(
            in_channels=out_channels[-1], out_channels=output_channels, 
            kernel_size=kernel_sizes[-1], stride=strides[-1], padding=0
        )
        self.final_activation = nn.Identity()

    def forward_method(self, x, *args):
        x = self.fc_layers(x)
        if self.reshape_size is not None:
            x = x.view(x.shape[0], *self.reshape_size)
        x = self.deconv_layers(x)
        x = self.final_deconv(x)
        x = self.final_activation(x)
        return x


class StaircaseConvolutionalDecoder(DecoderBase):
    def __init__(self, channels, kernels, strides, paddings, sequence = None):
        super(StaircaseConvolutionalDecoder, self).__init__(mean_first=False)

        # Channels includes the input one, so include that
        assert len(channels) - 1 == len(kernels) == len(strides) == len(paddings)

        if sequence is None:
            sequence = []
            [sequence.extend(['D', 'C']) for _ in range(len(kernels)//2)]
        else:
            assert len(sequence) == len(kernels)


        iterator = list(zip(channels[1:], kernels, strides, paddings, sequence))
        layers = []
        for i, (c, k, s, p, t) in enumerate(iterator):
            if t == 'D':
                new_layer = DeconvBlock(
                    in_channels=channels[i], out_channels=c, kernel_size=k, stride=s, concat_idx=-1, padding=p
                )
            elif t == 'C':
                new_layer = ConvBlock(
                    in_channels=channels[i], out_channels=c, kernel_size=k, stride=s, base_padding=p, padding=False
                )
            else:
                raise ValueError(t)
            layers.append(new_layer)

        self.layers = nn.Sequential(*layers)

    def forward_method(self, x):
        return self.layers(x)



if __name__ == '__main__':

    from classes_utils.architecture_integration import SkipEncoderDecoderEnsemble
    from classes_architectures.cifar.encoder import (
        DEFAULT_UNET_ENCODER_OUT_CHANNELS,
        DEFAULT_UNET_ENCODER_KERNEL_SIZES,
        DEFAULT_UNET_ENCODER_STRIDES,
        COMPRESSED_UNET_ENCODER_OUT_CHANNELS,
        COMPRESSED_UNET_ENCODER_KERNEL_SIZES,
        COMPRESSED_UNET_ENCODER_STRIDES,
        DEFAULT_NOSKIP_ENCODER_OUT_CHANNELS,
        DEFAULT_NOSKIP_ENCODER_KERNEL_SIZES,
        DEFAULT_NOSKIP_ENCODER_STRIDES
    )

    ensemble_type = 'basic'
    ensemble_size = 1
    
    if sys.argv[1] == 'unet':

        encoder_type = decoder_type = 'unet'
        
        encoder_ensemble_kwargs = {
            "input_size": (3, 32, 32),
            "out_channels": DEFAULT_UNET_ENCODER_OUT_CHANNELS,
            "kernel_sizes": DEFAULT_UNET_ENCODER_KERNEL_SIZES,
            "strides": DEFAULT_UNET_ENCODER_STRIDES, 
            "variational": False
        }

        decoder_ensemble_kwargs = {
            "embedding_dim": 256,
            "output_channels": 3,
            "out_channels": COMPRESSED_UNET_DECODER_OUT_CHANNELS,
            "kernel_sizes": COMPRESSED_UNET_DECODER_KERNEL_SIZES,
            "strides": COMPRESSED_UNET_DECODER_STRIDES, 
            "concat_idxs": COMPRESSED_UNET_DECODER_CONCATS,
        }

        autoencoder = SkipEncoderDecoderEnsemble(
            ensemble_type=ensemble_type,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            ensemble_size=ensemble_size,
            encoder_ensemble_kwargs=encoder_ensemble_kwargs,
            decoder_ensemble_kwargs=decoder_ensemble_kwargs
        )

    
    if sys.argv[1] == 'compressed_unet':

        encoder_type = decoder_type = 'unet'
        
        encoder_ensemble_kwargs = {
            "input_size": (3, 32, 32),
            "out_channels": COMPRESSED_UNET_ENCODER_OUT_CHANNELS,
            "kernel_sizes": COMPRESSED_UNET_ENCODER_KERNEL_SIZES,
            "strides": COMPRESSED_UNET_ENCODER_STRIDES, 
            "variational": False
        }

        decoder_ensemble_kwargs = {
            "embedding_dim": 256,
            "output_channels": 3,
            "out_channels": COMPRESSED_UNET_DECODER_OUT_CHANNELS,
            "kernel_sizes": COMPRESSED_UNET_DECODER_KERNEL_SIZES,
            "strides": COMPRESSED_UNET_DECODER_STRIDES, 
            "concat_idxs": COMPRESSED_UNET_DECODER_CONCATS,
        }

        autoencoder = SkipEncoderDecoderEnsemble(
            ensemble_type=ensemble_type,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            ensemble_size=ensemble_size,
            encoder_ensemble_kwargs=encoder_ensemble_kwargs,
            decoder_ensemble_kwargs=decoder_ensemble_kwargs
        )



    elif sys.argv[1] == 'no_skip':

        raise Exception

        encoder_type = decoder_type = 'no_skip'
        
        encoder_ensemble_kwargs = {
            "input_size": (3, 32, 32),
            "out_channels": DEFAULT_NOSKIP_ENCODER_OUT_CHANNELS,
            "kernel_sizes": DEFAULT_NOSKIP_ENCODER_KERNEL_SIZES,
            "strides": DEFAULT_NOSKIP_ENCODER_STRIDES, 
            "variational": False
        }

        decoder_ensemble_kwargs = {
            "embedding_dim": 1024,
            "output_channels": 3,
            "fc_sizes": DEFAULT_NOSKIP_FC_SIZES,
            "reshape_size": DEFAULT_NOSKIP_RESHAPE_SIZE,
            "out_channels": DEFAULT_NOSKIP_DECODER_OUT_CHANNELS,
            "kernel_sizes": DEFAULT_NOSKIP_DECODER_KERNEL_SIZES,
            "strides": DEFAULT_NOSKIP_DECODER_STRIDES
        }

        autoencoder = SkipEncoderDecoderEnsemble(
            ensemble_type=ensemble_type,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            ensemble_size=ensemble_size,
            encoder_ensemble_kwargs=encoder_ensemble_kwargs,
            decoder_ensemble_kwargs=decoder_ensemble_kwargs
        )

    print(autoencoder)

    for p in autoencoder.named_parameters():
        print(p[0], p[1].numel()/1e6)

    print("Total Parameters = ", sum(p.numel() for p in autoencoder.parameters())/1000000.0, "M")

    batch = torch.randn(16, 3, 32, 32)

    encodings, decodings = autoencoder(batch)

    print(encodings[0].shape, decodings[0].shape)
