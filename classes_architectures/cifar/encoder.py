import torch
from torch import nn
from torch.nn import functional as F
from util_functions.base import load_state_dict
from classes_architectures.base import ImageEncoderBase

# Architecture from https://codahead.com/blog/a-denoising-autoencoder-for-cifar-datasets

DEFAULT_UNET_ENCODER_OUT_CHANNELS=[64,128,256,256]
DEFAULT_UNET_ENCODER_KERNEL_SIZES=[3,3,3,3]
DEFAULT_UNET_ENCODER_STRIDES=[2,2,2,2]

DEFAULT_NOSKIP_ENCODER_OUT_CHANNELS=[16,32]
DEFAULT_NOSKIP_ENCODER_KERNEL_SIZES=[4,4]
DEFAULT_NOSKIP_ENCODER_STRIDES=[2,2]


__all__ = [
    'UNetEncoder',
    'NoSkipEncoder'
]


class EncoderPadding(nn.Module):
    def __init__(self, kernel_size, stride):
        super(EncoderPadding, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.kernel_size = kernel_size
        self.base_padding = kernel_size // 2
        self.stride = stride

    @staticmethod
    def get_left_padding(remain):
        return remain // 2
    
    @staticmethod
    def get_right_padding(remain):
        return remain // 2 + (remain % 2)
        
    def forward(self, x):
        
        # First pad based on kernel size
        x = F.pad(x, (self.base_padding, ) * 4, mode = 'constant')
        
        # Get sequence length to ensure mapping is reversible
        length = x.size(-2)
        
        # This is to ensure sequence length is always even
        remain = length % self.stride
        
        # left, right, top, bottom
        leftpad = self.get_left_padding(remain)
        rightpad = self.get_right_padding(remain)
                
        return F.pad(x, (0, 0, leftpad, rightpad), mode = 'constant')


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, base_padding=0, padding = True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, stride=stride,
            padding=base_padding
        )
        self.pad = EncoderPadding(kernel_size=kernel_size, stride=stride) if padding else nn.Identity()
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UNetEncoder(ImageEncoderBase):

    def __init__(self, input_size, out_channels, kernel_sizes, strides, variational, *args, flatten = False, **kwargs):

        assert len(input_size) == 3, 'input_size must be three dimensional'
        input_channels = input_size[0]
        layers = [ConvBlock(input_channels, out_channels[0], kernel_sizes[0], strides[0])]
        for i in range(1, len(out_channels)):
            layers.append(ConvBlock(out_channels[i-1], out_channels[i], kernel_sizes[i], strides[i]))
        # layers[-1].padding = False
        if flatten:
            layers.append(nn.Flatten())
        layers = nn.ModuleList(layers)

        embedding_dim = out_channels[-1]
        dropout_idxs = []
        noise_idx = 0

        super(UNetEncoder, self).__init__(
            input_size, layers, dropout_idxs, noise_idx, variational, embedding_dim, return_skips=True
        )


class NoSkipEncoder(ImageEncoderBase):

    def __init__(
        self, input_size, fc_sizes, out_channels, kernel_sizes, strides, variational, base_padding, sequence_padding,
        *args, flatten = False, **kwargs
    ):

        layers = [ConvBlock(input_size[0], out_channels[0], kernel_sizes[0], strides[0], base_padding, sequence_padding)]
        for i in range(1, len(out_channels)):
            layers.append(
                ConvBlock(out_channels[i-1], out_channels[i], kernel_sizes[i], strides[i], base_padding, sequence_padding)
            )
        noise_idx = len(layers)

        if len(fc_sizes) or flatten:
            layers.append(nn.Flatten())

        for i, fcs in enumerate(fc_sizes[1:]):
            layers.append(nn.Linear(fc_sizes[i], fcs))
            layers.append(nn.Tanh())

        if variational:
            assert len(fc_sizes), "For VAE need at least one FC layer"
        embedding_dim = fc_sizes[-1] if len(fc_sizes) else 0

        layers = nn.ModuleList(layers)

        super(NoSkipEncoder, self).__init__(
            input_size, layers, dropout_idxs=[], noise_idx=noise_idx, variational=variational, 
            embedding_dim=embedding_dim, return_skips=False
        )


if __name__ == '__main__':

    encoder = UNetEncoder(
        (3, 32, 32), DEFAULT_UNET_ENCODER_OUT_CHANNELS, DEFAULT_UNET_ENCODER_KERNEL_SIZES, 
        DEFAULT_UNET_ENCODER_STRIDES, False
    )

    print(encoder)

    batch = torch.randn(32, 3, 32, 32)

    encoding, skip_list = encoder(batch)

    for sl in skip_list:
        print(sl.shape)
    print(encoding.shape)
