import torch
from torch import nn
from cifar_repo.models.cifar import densenet, resnet
from classes_architectures.audio.decoder import FCDecoder
from classes_utils.layers import EmbeddingLoadingLayer
from config.ootb.creation_functions import make_unet_regression_architecture
from classes_architectures.cifar.encoder import DEFAULT_UNET_ENCODER_KERNEL_SIZES, DEFAULT_UNET_ENCODER_OUT_CHANNELS, DEFAULT_UNET_ENCODER_STRIDES


class InterfaceFriendlyModel(nn.Module):
    def __init__(self, model):
        super(InterfaceFriendlyModel, self).__init__()
        self.model = model
        self.muzzle = nn.Identity()
    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return ([output], [self.muzzle(output)])


def default_simple_cifar_convolutional_classifier(dropout, regression_mode = False):
    return make_unet_regression_architecture(
        encoder_out_channel_list=DEFAULT_UNET_ENCODER_OUT_CHANNELS,
        encoder_kernel_size_list=DEFAULT_UNET_ENCODER_KERNEL_SIZES,
        encoder_stride_list=DEFAULT_UNET_ENCODER_STRIDES,

        # Logits always output
        decoder_nonlinearities=['relu', 'relu', 'none'],
        dropout=dropout,
        input_size=(3, 32, 32),
        embedding_dim=1024,
        decoder_layer_dims=[256, 64, 1 if regression_mode else 2],
        mult_noise=0,
        variational=False
    )


def default_mini_resnet_classifier(regression_mode = False, *args, **kwargs):
    model = resnet(depth=8, num_classes=1 if regression_mode else 2)
    print(f'Total miniature resnet DAF params: {(sum(p.numel() for p in model.parameters()))}')
    return InterfaceFriendlyModel(model)


def default_mini_densenet_classifier(depth, *args, dropout=0., regression_mode=False, num_classes=None, **kwargs):
    # num_classes would only be required for actually posterior distillation/multitask setting
    # Otherwise, a single number to represent acq/p(labelled) is sufficient
    if num_classes == None:
        model = densenet(num_classes=1 if regression_mode else 2, depth=depth, growthRate=12, dropRate=dropout)
    else:
        model = densenet(num_classes=num_classes, depth=depth, growthRate=12, dropRate=dropout)
    print(f'Total miniature densenet DAF params: {(sum(p.numel() for p in model.parameters()))}')
    return InterfaceFriendlyModel(model)


def byol_binary_linear_classification(embedding_cache_path, regression_mode=False, num_classes=None):
    # See default_mini_densenet_classifier for the logic here
    layer_dim = (1 if regression_mode else 2) if num_classes==None else num_classes
    
    # To make it work for InterfaceFriendlyModel, which is required for muzzle customisation
    model = nn.Sequential(
        # Embedding loading layer
        EmbeddingLoadingLayer(embedding_cache_path),
        # Output logits of required size, i.e. no non-linearity
        nn.Linear(2048, layer_dim)
    )

    print("num parameters don't apply for byol_binary_linear_classification")
    return InterfaceFriendlyModel(model)


if __name__ == '__main__':

    import torch
    from torch import nn
    from torch import optim
    from training_scripts.cifar_daf_scripts import train_daf_labelled_classification


    batch_size = 42
    num_channels = 3
    image_dim = 32

    data = torch.randn(batch_size, num_channels, image_dim, image_dim)
    labels = (torch.FloatTensor(batch_size).uniform_() > 0.5).int().long()

    model = default_simple_cifar_convolutional_classifier(0, use_logits=True)
    output = model(data)

    train_daf_labelled_classification(
        classifier=model,
        optimizer=optim.SGD(model.parameters(), lr=0.001),
        scheduler=None,
        scheduler_epochs=[],
        encodings_criterion=None,
        decodings_criterion=nn.CrossEntropyLoss(reduction='mean'),
        anchor_criterion=None,
        train_dataloader=[(data, labels)],
        test_dataloader=[],
        num_epochs=3,
    )
