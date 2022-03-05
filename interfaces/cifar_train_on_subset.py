import torch, random, argparse, os, json
import torch.backends.cudnn as cudnn
from torch.utils import data
from torch import nn
import torch.optim as optim
import cifar_repo.models.cifar as models
from cifar_repo.utils import Logger, savefig
from training_scripts.cifar_active_learning import make_ensemble, ModelWrapper
from cifar_repo.cifar import (
    transform_train, transform_test, train, test, adjust_learning_rate
)
from classes_utils.cifar.data import CIFAR100Subset, CIFAR100, CIFAR10Subset, CIFAR10
from util_functions.base import torch_manual_script


# Use CUDA
use_cuda = torch.cuda.is_available()    

parser = argparse.ArgumentParser()

parser.add_argument(
    '--subset_size', type = int, 
    help="The size of the subset being trained, for verification and naming purposes"
)
parser.add_argument(
    '--subset_index_path', type = str, 
    help="Where the index txt file is, e.g. /home/alta/BLTSpeaking/exp-pr450/logs/cifar_daf_active_learning/round_history-13/indices_2501.txt"
)
parser.add_argument(
    '--log_base', type = str, 
    help="Where this round is checkpointed"
)
parser.add_argument(
    '--mirrored_config_json_path', type = str, 
    help="Where the original config is, for model/dataset configuration, e.g. /home/alta/BLTSpeaking/exp-pr450/logs/cifar_daf_active_learning/round_history-13/mirrored_config.json"
)

script_args = parser.parse_args()


class Config:
    def __init__(self, d):
        self.__dict__.update(d)
    def update(self, d):
        self.__dict__.update(d)


def train_cifar_subset_main(subset_indices, mirrored_args, state):

    start_epoch = mirrored_args.start_epoch  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing dataset %s' % mirrored_args.dataset)
    if mirrored_args.dataset == 'cifar10':
        train_dataset, test_dataset, num_classes = CIFAR10Subset, CIFAR10, 10
        print('Using cifar10')
    else:
        train_dataset, test_dataset, num_classes = CIFAR100Subset, CIFAR100, 100
        print('Using cifar100')

    print(len(subset_indices), 'labelled', flush = True)
    trainset = train_dataset(
        root='./data', train=True, download=True, transform=transform_train, original_transform=None, target_transform=None,
        init_indices=subset_indices
    )
    trainloader = data.DataLoader(trainset, batch_size=mirrored_args.train_batch, shuffle=True, num_workers=mirrored_args.workers)

    testset = test_dataset(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=mirrored_args.test_batch, shuffle=False, num_workers=mirrored_args.workers)

    # ModeL
    model = ModelWrapper()
    model.reinit_model(make_ensemble(mirrored_args, 1, num_classes))
    model.midloop = True
    
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=mirrored_args.lr, momentum=mirrored_args.momentum, weight_decay=mirrored_args.weight_decay)

    # Resume
    title = 'cifar-10-' + mirrored_args.arch
    if mirrored_args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(mirrored_args.resume), 'Error: no checkpoint directory found!'
        mirrored_args.log_base = os.path.dirname(mirrored_args.resume)
        checkpoint = torch.load(mirrored_args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(mirrored_args.log_base, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(mirrored_args.log_base, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if mirrored_args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, mirrored_args.epochs):
        adjust_learning_rate(optimizer, epoch, mirrored_args, state)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, mirrored_args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        print(f'train_loss: {train_loss} | train_acc: {train_acc} | test_loss: {test_loss} | test_acc: {test_acc}')

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

    logger.close()
    logger.plot()
    # savefig(os.path.join(mirrored_args.checkpoint, 'log.eps'))

    return model


if __name__ == '__main__':

    # Args passed to this
    main_args = parser.parse_args()
    args = Config(vars(main_args))

    # Collect args from mirrord config, for model construction
    with open(args.mirrored_config_json_path, 'r') as jf:
        args.update(json.load(jf))

    args_state = args.__dict__

    # Validate dataset
    assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

    # Random seed
    torch_manual_script(args)

    with open(args.subset_index_path, 'r') as f:
        indices = f.read().split('\n')[:-1]
        assert len(indices) == args.subset_size
    subset_indices = list(map(lambda x: int(x), indices))

    train_cifar_subset_main(subset_indices, args, args_state)
