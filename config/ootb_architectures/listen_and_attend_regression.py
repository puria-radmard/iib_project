from config.ootb_architectures.creation_functions import (
    make_blstm_listener_self_attention_regression_architecture,
    make_blstm_listener_transformer_regression_architecture
)


def default_blstm_listener_self_attention_regression_architecture(dropout, use_logits, classification=True):
    if classification:
        act = 'none' if use_logits else 'softmax'
    else:
        act = 'sigmoid'
    return make_blstm_listener_self_attention_regression_architecture(
        lstm_hidden_size=128,
        pyramid_size=3,
        key_size=32,
        value_size=32,
        query_size=32,
        num_heads=8,
        decoder_layer_dims=[2 if classification else 1],
        decoder_nonlinearities=[act],
        dropout=dropout,
        mfcc_dim=40
    )


def default_blstm_listener_transformer_regression_architecture(dropout, use_logits, classification=True):
    if classification:
        act = 'none' if use_logits else 'softmax'
    else:
        act = 'sigmoid'
    return make_blstm_listener_transformer_regression_architecture(
        pyramid_size=3,
        d_model=256,
        num_heads=8,
        hidden_sizes=[2048],
        num_attn_blocks=2,
        decoder_layer_dims=[2 if classification else 1],
        decoder_nonlinearities=[act],
        dropout=dropout,
        mfcc_dim=40
    )



if __name__ == '__main__':

    import torch
    from torch import nn
    from torch import optim
    from training_scripts.audio_regression_scripts import audio_regression_script

    batch_size = 4
    sequence_length = 4000
    feature_dim = 40

    data = torch.randn(batch_size, sequence_length, feature_dim)

    model = default_blstm_listener_transformer_regression_architecture(0, use_logits=True)
    output = model(data)
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
