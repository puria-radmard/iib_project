from torch import nn
from cifar_repo.models.cifar import densenet, resnet
from config.ootb_architectures.creation_functions import make_unet_regression_architecture
from classes_architectures.cifar.encoder import DEFAULT_UNET_ENCODER_KERNEL_SIZES, DEFAULT_UNET_ENCODER_OUT_CHANNELS, DEFAULT_UNET_ENCODER_STRIDES

class InterfaceFriendlyModel(nn.Module):
    def __init__(self, model):
        self.model = model
    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return ([None], [output])


def default_simple_cifar_convolutional_classifier(dropout, use_logits):
    return make_unet_regression_architecture(
        encoder_out_channel_list=DEFAULT_UNET_ENCODER_OUT_CHANNELS,
        encoder_kernel_size_list=DEFAULT_UNET_ENCODER_KERNEL_SIZES,
        encoder_stride_list=DEFAULT_UNET_ENCODER_STRIDES,
        decoder_nonlinearities=['tanh', 'tanh', 'none' if use_logits else 'softmax'],
        dropout=dropout,
        input_size=(3, 32, 32),
        embedding_dim=1024, # FIX THIS
        decoder_layer_dims=[256, 64, 2],
        mult_noise=0,
        variational=False
    )


def default_mini_resnet_classifier(*args, **kwargs):
    model = resnet(depth=0)
    print('Total miniature resnet DAF params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    return InterfaceFriendlyModel(model)


def default_mini_densenet_classifier(dropout, *args, **kwargs):
    model = densenet(num_classes=2, depth=4, growthRate=12, dropRate=dropout)
    print('Total miniature densenet DAF params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
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
