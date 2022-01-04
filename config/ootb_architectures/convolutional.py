from classes_architectures.cifar.decoder import (
    DEFAULT_UNET_DECODER_OUT_CHANNELS, DEFAULT_UNET_DECODER_KERNEL_SIZES, 
    DEFAULT_UNET_DECODER_STRIDES, DEFAULT_UNET_DECODER_CONCATS,
)
from classes_architectures.cifar.encoder import DEFAULT_UNET_ENCODER_KERNEL_SIZES, DEFAULT_UNET_ENCODER_OUT_CHANNELS, DEFAULT_UNET_ENCODER_STRIDES
from config.ootb_architectures.creation_functions import make_unet_architecture, make_staircase_autoencoder_architecture


def default_unet_network():
    return make_unet_architecture(
        encoder_out_channel_list=DEFAULT_UNET_ENCODER_OUT_CHANNELS, 
        encoder_kernel_size_list=DEFAULT_UNET_ENCODER_KERNEL_SIZES, 
        encoder_stride_list=DEFAULT_UNET_ENCODER_STRIDES,
        decoder_out_channel_list=DEFAULT_UNET_DECODER_OUT_CHANNELS,
        decoder_kernel_size_list=DEFAULT_UNET_DECODER_KERNEL_SIZES,
        decoder_stride_list=DEFAULT_UNET_DECODER_STRIDES, 
        input_size=(3, 32, 32), 
        embedding_dim=256, 
        decoder_concat_list=DEFAULT_UNET_DECODER_CONCATS, 
        output_channels=3,
    )


def no_skip_default_unet_network():
    return make_unet_architecture(
        encoder_out_channel_list=DEFAULT_UNET_ENCODER_OUT_CHANNELS, 
        encoder_kernel_size_list=DEFAULT_UNET_ENCODER_KERNEL_SIZES, 
        encoder_stride_list=DEFAULT_UNET_ENCODER_STRIDES,
        decoder_out_channel_list=DEFAULT_UNET_DECODER_OUT_CHANNELS,
        decoder_kernel_size_list=DEFAULT_UNET_DECODER_KERNEL_SIZES,
        decoder_stride_list=DEFAULT_UNET_DECODER_STRIDES, 
        input_size=(3, 32, 32), 
        embedding_dim=256, 
        decoder_concat_list=DEFAULT_UNET_DECODER_CONCATS, 
        output_channels=3,
    )


def default_staircase_network():
    return make_staircase_autoencoder_architecture(
        encoder_out_channel_lis=[16, 24, 128],
        encoder_kernel_size_list=[3, 3, 3],
        encoder_stride_list=[2, 2, 2],
        encoder_fc_sizes=[],
        decoder_out_channel_list=[128, 32, 32, 16, 16, 3, 3],
        decoder_kernel_size_list=[3, 2, 3, 2, 3, 2],
        decoder_stride_list=[2, 1, 2, 1, 2, 1],
        decoder_padding_list=[0, 0, 0, 0, 0, 0],
        decoding_operation_sequence="DCDCDC",
        input_size=(3, 32, 32), sequence_padding=False, base_padding=1, 
        mult_noise=0, ensemble_size=1, variational=False
    )


def default_audio_unet_network():
    return make_unet_architecture(
        encoder_out_channel_list=DEFAULT_UNET_ENCODER_OUT_CHANNELS[:-1],
        encoder_kernel_size_list=DEFAULT_UNET_ENCODER_KERNEL_SIZES[:-1],
        encoder_stride_list=DEFAULT_UNET_ENCODER_STRIDES[:-1],
        decoder_out_channel_list=DEFAULT_UNET_DECODER_OUT_CHANNELS[1:], #[128, 64, 32],
        decoder_kernel_size_list=DEFAULT_UNET_DECODER_KERNEL_SIZES[1:], #[(4, 3), 3, 3, 4],
        decoder_stride_list=DEFAULT_UNET_DECODER_STRIDES[1:], #[2, 2, 2, 1], 
        input_size=(1, None, 40),
        embedding_dim=256,
        decoder_concat_list=[-1, 2, 1],
        output_channels=1,
    )


def default_noskip_audio_unet_network():
    return make_unet_architecture(
        encoder_out_channel_list=DEFAULT_UNET_ENCODER_OUT_CHANNELS[:-1],
        encoder_kernel_size_list=DEFAULT_UNET_ENCODER_KERNEL_SIZES[:-1],
        encoder_stride_list=DEFAULT_UNET_ENCODER_STRIDES[:-1],
        decoder_out_channel_list=DEFAULT_UNET_DECODER_OUT_CHANNELS[1:], #[128, 64, 32],
        decoder_kernel_size_list=DEFAULT_UNET_DECODER_KERNEL_SIZES[1:], #[(4, 3), 3, 3, 4],
        decoder_stride_list=DEFAULT_UNET_DECODER_STRIDES[1:], #[2, 2, 2, 1], 
        input_size=(1, None, 40),
        embedding_dim=256,
        decoder_concat_list=[-1, -1, -1],
        output_channels=1,
    )

