
from config.ootb_architectures.creation_functions import make_listen_and_attend_lstm_regression_architecture


def default_listen_and_attend_lstm_regression_network(decoder_dropout, use_logits):
    return make_listen_and_attend_lstm_regression_architecture(
        lstm_hidden_size=256,
        pyramid_size=3,
        key_size=512,
        value_size=512,
        query_size=512,
        num_heads=6,
        num_attn=3,
        decoder_layer_dims=[256, 256, 64, 2],
        decoder_nonlinearities=['tanh', 'tanh', 'tanh', 'none' if use_logits else 'softmax'],
        decoder_dropout=decoder_dropout,
        mfcc_dim=40
    )