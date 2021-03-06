from classes_utils.architecture.architecture_integration import (
    AudioEncoderDecoderEnsemble, EncoderDecoderEnsemble, SkipEncoderDecoderEnsemble
)


def make_recurrent_regression_architecture(
        cell_type, encoder_lstm_layers, encoder_lstm_sizes, decoder_fc_hidden_dims, 
        dropout, embedding_dim, device, feature_dim=40, ensemble_type='basic',
        variational = False
    ):
    
    encoder_ensemble_kwargs = {
        "mfcc_dim": feature_dim, "embedding_dim": embedding_dim, 
        "dropout_rate": dropout, "variational": variational, 
        "lstm_sizes": encoder_lstm_sizes,
        "lstm_layers": encoder_lstm_layers,
        "fc_hidden_dims": [],
        "cell_type": cell_type
    } 
    decoder_ensemble_kwargs = {
        "embedding_dim": embedding_dim,
        "layer_dims": decoder_fc_hidden_dims,
        # Need sigmoid for regression
        "nonlinearities": ['tanh' for _ in decoder_fc_hidden_dims[:-1]] + ['sigmoid'],
        "dropout_rate": dropout, "mean_first": False,
    }

    ensemble = AudioEncoderDecoderEnsemble(
        ensemble_type, "BidirectionalLSTMAudioEncoder", "FCDecoder",
        1, encoder_ensemble_kwargs, decoder_ensemble_kwargs
    ).to(device)

    return ensemble


def make_unet_architecture(
        encoder_out_channel_list, encoder_kernel_size_list, encoder_stride_list,
        decoder_out_channel_list, decoder_kernel_size_list, decoder_stride_list, 
        input_size, embedding_dim, decoder_concat_list, output_channels,
        mult_noise=0, variational=False
    ):

    encoder_ensemble_kwargs = {
        "input_size": input_size,
        "out_channels": encoder_out_channel_list,
        "kernel_sizes": encoder_kernel_size_list,
        "strides": encoder_stride_list, 
        "variational": variational
    }

    decoder_ensemble_kwargs = {
        "embedding_dim": embedding_dim,
        "output_channels": output_channels,
        "out_channels": decoder_out_channel_list,
        "kernel_sizes": decoder_kernel_size_list,
        "strides": decoder_stride_list, 
        "concat_idxs": decoder_concat_list,
    }

    autoencoder_ensemble = SkipEncoderDecoderEnsemble(
        ensemble_type='basic',
        encoder_type='UNetEncoder',
        decoder_type='UNetDecoder',
        ensemble_size=1,
        encoder_ensemble_kwargs=encoder_ensemble_kwargs,
        decoder_ensemble_kwargs=decoder_ensemble_kwargs,
        mult_noise=mult_noise
    )

    return autoencoder_ensemble


def make_unet_regression_architecture(
        encoder_out_channel_list, encoder_kernel_size_list, encoder_stride_list,
        decoder_nonlinearities, dropout, input_size, embedding_dim, decoder_layer_dims,
        mult_noise=0, variational=False
    ):

    encoder_ensemble_kwargs = {
        "input_size": input_size,
        "out_channels": encoder_out_channel_list,
        "kernel_sizes": encoder_kernel_size_list,
        "strides": encoder_stride_list, 
        "variational": variational,
        "flatten": True
    }

    decoder_ensemble_kwargs = {
        "embedding_dim": embedding_dim,
        "layer_dims": decoder_layer_dims,
        "nonlinearities": decoder_nonlinearities,
        "dropout_rate": dropout,
        "mean_first": False,
    }

    autoencoder_ensemble = SkipEncoderDecoderEnsemble(
        ensemble_type='basic',
        encoder_type='UNetEncoder',
        decoder_type='FCDecoder',
        ensemble_size=1,
        encoder_ensemble_kwargs=encoder_ensemble_kwargs,
        decoder_ensemble_kwargs=decoder_ensemble_kwargs,
        mult_noise=mult_noise
    )

    return autoencoder_ensemble


def make_staircase_autoencoder_architecture(
        encoder_out_channel_list, encoder_kernel_size_list, encoder_stride_list,
        encoder_fc_sizes, decoder_out_channel_list, decoder_kernel_size_list, 
        decoder_stride_list, decoder_padding_list, decoding_operation_sequence, input_size, 
        sequence_padding, base_padding=1, mult_noise=0, ensemble_size=1, variational=False
    ):

    encoder_ensemble_kwargs = {
        "input_size": input_size,
        "fc_sizes": encoder_fc_sizes,
        "out_channels": encoder_out_channel_list,
        "kernel_sizes": encoder_kernel_size_list,
        "strides": encoder_stride_list,
        "variational": variational,
        "base_padding": base_padding,
        "sequence_padding": sequence_padding
    }

    decoder_ensemble_kwargs = {
        "channels": decoder_out_channel_list,
        "kernels": decoder_kernel_size_list,
        "strides": decoder_stride_list,
        "paddings": decoder_padding_list,
        "sequence": decoding_operation_sequence
    }

    autoencoder_ensemble = EncoderDecoderEnsemble(
        ensemble_type='basic',
        encoder_type='NoSkipEncoder',
        decoder_type='StaircaseConvolutionalDecoder',
        ensemble_size=ensemble_size,
        encoder_ensemble_kwargs=encoder_ensemble_kwargs,
        decoder_ensemble_kwargs=decoder_ensemble_kwargs,
        mult_noise=mult_noise
    )

    return autoencoder_ensemble


def make_listener_self_attention_regression_architecture(
        encoder_type, key_size, value_size, query_size, num_heads,
        decoder_layer_dims, decoder_nonlinearities, dropout, 
        mfcc_dim=40, **listener_kwargs
    ):

    encoder_ensemble_kwargs = {
        "mfcc_dim": mfcc_dim,
        "key_size": key_size,
        "query_size": query_size,
        "value_size": value_size,
        "num_heads": num_heads,
        "dropout": dropout,
        "variational": False,
        **listener_kwargs
    }

    decoder_ensemble_kwargs = {
        "embedding_dim": value_size*num_heads, 
        "layer_dims": decoder_layer_dims, 
        "nonlinearities": decoder_nonlinearities, 
        "dropout_rate": dropout, 
        "mean_first": False
    }

    autoencoder_ensemble = EncoderDecoderEnsemble(
        ensemble_type='basic',
        encoder_type=encoder_type,
        decoder_type='FCDecoder',
        ensemble_size=1,
        encoder_ensemble_kwargs=encoder_ensemble_kwargs,
        decoder_ensemble_kwargs=decoder_ensemble_kwargs,
        mult_noise=0
    )

    return autoencoder_ensemble


def make_listener_transformer_regression_architecture(
        encoder_type, d_model, num_heads, hidden_sizes, num_attn_blocks, 
        decoder_layer_dims, decoder_nonlinearities, dropout, aggregation,
        mfcc_dim=40, **listener_kwargs
    ):

    encoder_ensemble_kwargs = {
        "mfcc_dim": mfcc_dim,
        "d_model": d_model,
        "num_heads": num_heads,
        "hidden_sizes": hidden_sizes,
        "num_attn_blocks": num_attn_blocks,
        "transformer_aggregation": aggregation,
        "dropout": dropout,
        "variational": False,
        **listener_kwargs
    }

    decoder_ensemble_kwargs = {
        "embedding_dim": d_model, 
        "layer_dims": decoder_layer_dims, 
        "nonlinearities": decoder_nonlinearities, 
        "dropout_rate": dropout, 
        "mean_first": False
    }

    autoencoder_ensemble = EncoderDecoderEnsemble(
        ensemble_type='basic',
        encoder_type=encoder_type,
        decoder_type='FCDecoder',
        ensemble_size=1,
        encoder_ensemble_kwargs=encoder_ensemble_kwargs,
        decoder_ensemble_kwargs=decoder_ensemble_kwargs,
        mult_noise=0
    )

    return autoencoder_ensemble


def make_simple__architecture(
        cell_type, encoder_lstm_layers, encoder_lstm_sizes, decoder_fc_hidden_dims, 
        dropout, embedding_dim, device, feature_dim=40, ensemble_type='basic',
        variational = False
    ):
    
    encoder_ensemble_kwargs = {
        "mfcc_dim": feature_dim, "embedding_dim": embedding_dim, 
        "dropout_rate": dropout, "variational": variational, 
        "lstm_sizes": encoder_lstm_sizes,
        "lstm_layers": encoder_lstm_layers,
        "fc_hidden_dims": [],
        "cell_type": cell_type
    } 
    decoder_ensemble_kwargs = {
        "embedding_dim": embedding_dim,
        "layer_dims": decoder_fc_hidden_dims,
        # Need sigmoid for regression
        "nonlinearities": ['tanh' for _ in decoder_fc_hidden_dims[:-1]] + ['sigmoid'],
        "dropout_rate": dropout, "mean_first": False,
    }

    ensemble = AudioEncoderDecoderEnsemble(
        ensemble_type, "BidirectionalLSTMAudioEncoder", "FCDecoder",
        1, encoder_ensemble_kwargs, decoder_ensemble_kwargs
    ).to(device)

    return ensemble
