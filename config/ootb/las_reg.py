import sys
from config.ootb.creation_functions import (
    make_listener_self_attention_regression_architecture,
    make_listener_transformer_regression_architecture
)

DEFAULT_MOVING_DNN_LISTENER_KWARGS = {
    "listener_num_frames": [30, 15],
    "listener_strides": [25, 5],
    "listener_hidden_dims": [
        [256, 256],
        [256, 256]
    ],
    "listener_non_lin_funcs": ['relu', 'relu'],
}

DEFAULT_TDNN_LISTENER_KWARGS = {
    "listener_output_dims": [128, 512, 256],
    "listener_context_idxs": [[-2, -1, 0, 1, 2], [-4, 0, 4], [-7, 0, 7]],
    "listener_non_lin_funcs": ['relu', 'relu', 'relu'],
    "listener_strides": [3, 1, 1]
}


def _default_self_attention_regression_architecture(encoder_type, listener_kwargs, dropout, head_size):
    
    # Determine number of heads at the end of all the attention stuff
    head_size = head_size

    return make_listener_self_attention_regression_architecture(
        encoder_type=encoder_type,
        key_size=32,
        value_size=32,
        query_size=32,
        num_heads=8,
        decoder_layer_dims=[head_size],

        # Always using logits now, even for classification
        # This is because in the audio_uncertinaty_regression, we define out own
        # `muzzle` for the model, depending on the task
        decoder_nonlinearities=['none'],
        dropout=dropout,
        mfcc_dim=40,
        **listener_kwargs
    )


def _default_transformer_regression_architecture(encoder_type, listener_kwargs, dropout, head_size, aggregation):

    # Determine number of heads at the end of all the attention stuff
    head_size = head_size

    return make_listener_transformer_regression_architecture(
        encoder_type=encoder_type,
        d_model=256,
        num_heads=8,
        hidden_sizes=[2048],
        num_attn_blocks=2,
        decoder_layer_dims=[head_size],

        # Always using logits now, even for classification
        # This is because in the audio_uncertinaty_regression, we define out own
        # `muzzle` for the model, depending on the task
        decoder_nonlinearities=['none'],
        dropout=dropout,
        mfcc_dim=40,
        aggregation=aggregation,
        **listener_kwargs
    )


def default_blstm_listener_self_attention_regression_architecture(dropout, head_size):
    listener_kwargs = {"lstm_hidden_size": 128, "pyramid_size": 3}
    encoder_type = 'BLSTMListenerSelfAttentionEncoder'
    return _default_self_attention_regression_architecture(encoder_type, listener_kwargs, dropout, head_size)


def default_blstm_listener_transformer_regression_architecture(dropout, head_size, aggregation='mean'):
    listener_kwargs = {"pyramid_size": 3}
    encoder_type = 'BLSTMListenerTransformerEncoder'
    return _default_transformer_regression_architecture(encoder_type, listener_kwargs, dropout, head_size, aggregation)


def default_moving_dnn_listener_self_attention_regression_architecture(dropout, head_size):
    listener_kwargs = DEFAULT_MOVING_DNN_LISTENER_KWARGS
    encoder_type = 'SlidingDNNListenerSelfAttentionEncoder'
    return _default_self_attention_regression_architecture(encoder_type, listener_kwargs, dropout, head_size)


def default_moving_dnn_listener_transformer_regression_architecture(dropout, head_size, aggregation='mean'):
    listener_kwargs = DEFAULT_MOVING_DNN_LISTENER_KWARGS
    encoder_type = 'SlidingDNNListenerTransformerEncoder'
    return _default_transformer_regression_architecture(encoder_type, listener_kwargs, dropout, head_size, aggregation)


def default_tdnn_listener_self_attention_regression_architecture(dropout, head_size):
    listener_kwargs = DEFAULT_TDNN_LISTENER_KWARGS
    encoder_type = 'TDNNListenerSelfAttentionEncoder'
    return _default_self_attention_regression_architecture(encoder_type, listener_kwargs, dropout, head_size)


def default_tdnn_listener_transformer_regression_architecture(dropout, head_size, aggregation='mean'):
    listener_kwargs = DEFAULT_TDNN_LISTENER_KWARGS
    encoder_type = 'TDNNListenerTransformerEncoder'
    return _default_transformer_regression_architecture(encoder_type, listener_kwargs, dropout, head_size, aggregation)


if __name__ == '__main__':

    import torch
    from torch import nn
    from torch import optim
    from training_scripts.audio_regression_scripts import audio_regression_script

    batch_size = 4
    sequence_length = 60
    feature_dim = 40

    data = torch.randn(batch_size, sequence_length, feature_dim)

    model = globals()[sys.argv[1]](0, use_logits=True)
    output = model(data)

    print(output[0][0].shape, output[1][0].shape)
    print(sum(p.numel() for p in model.parameters())/1000000.0, "M parameters")
    import pdb; pdb.set_trace()

    audio_regression_script(
        ensemble=model,
        optimizer=optim.SGD(model.parameters(), lr=0.001),
        scheduler=None,
        scheduler_epochs=[],
        criterion=nn.CrossEntropyLoss(reduction='mean'),
        train_dataloader=[{"padded_features": data, "labelled": torch.tensor([0, 1, 1, 0])}],
        test_dataloader=[],
        num_epochs=1,
        is_regression=False,
        target_attribute_name='labelled',
        show_print=True,
    )
