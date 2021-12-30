from torch import optim
from classes_utils.layers import BidirectionalLSTMHiddenStateStacker
from config import DEFAULT_DTYPE
from training_scripts.audio_regression_scripts import audio_regression_script


if __name__ == '__main__':

    import torch
    from torch import nn
    from classes_utils.audio.las.attention import SelfAttention
    from config.ootb_architectures.listen_and_attend import default_listen_and_attend_lstm_regression_network

    batch_size = 4
    sequence_length = 4000
    feature_dim = 40

    data = torch.randn(batch_size, sequence_length, feature_dim)

    model = default_listen_and_attend_lstm_regression_network(0, use_logits=True)

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

    import pdb; pdb.set_trace()
