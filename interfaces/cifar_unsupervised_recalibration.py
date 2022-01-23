import os, argparse, torch, random
from torch import nn

from cifar_repo.cifar import transform_test

import active_learning as al

from classes_utils.cifar.data import CIFAR100LabelledClassification, CIFAR100Subset, CIFAR10LabelledClassification, CIFAR10Subset
from config.ootb import default_staircase_network, default_unet_network, no_skip_default_unet_network
from training_scripts.recalibration_scripts import unsupervised_recalibration_script
from classes_utils.base.model import ModelWrapper
from classes_losses.reconstrution import ImageReconstructionLoss
from util_functions.base import config_savedir
from torch.utils.data import DataLoader
from config import metric_functions
from util_functions.data import *


device = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu')
)
use_cuda = torch.cuda.is_available()

al.disable_tqdm()


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--metric_function", required=True, type=str)
parser.add_argument("--minibatch_prop", required=True, type=float)
parser.add_argument("--total_added_prop", required=True, type=float)
parser.add_argument("--total_start_prop", required=True, type=float)
parser.add_argument("--initial_lr", required=True, type=float, help="Learning rate of RAE optimizer")
parser.add_argument("--finetune_lr", required=True, type=float, help="Learning rate of RAE optimizer")
parser.add_argument("--scheduler_epochs", required=True, nargs="+", type=int)
parser.add_argument("--scheduler_proportion", required=True, type=float)
parser.add_argument("--dropout", required=True, type=float)
parser.add_argument("--weight_decay",required=True,type=float)
parser.add_argument("--finetune_weight_decay",required=True,type=float)
parser.add_argument('--unet_with_skips', dest='unet_skips', action='store_true')
parser.add_argument('--unet_without_skips', dest='unet_skips', action='store_false')
parser.set_defaults(unet_skips=True)
parser.add_argument('--compressed_net', action='store_true', default = False)
parser.add_argument('--data_aug_type', required=False, type=str, default="none")
parser.add_argument("--batch_size", required=True, type=int)
parser.add_argument("--num_initial_epochs", required=True, type=int)
parser.add_argument("--num_finetune_epochs", required=True, type=int)
parser.add_argument('--do_reinitialise_autoencoder_ensemble', dest='reinitialise_autoencoder_ensemble', action='store_true')
parser.add_argument('--do_not_reinitialise_autoencoder_ensemble', dest='reinitialise_autoencoder_ensemble', action='store_false')
parser.set_defaults(reinitialise_autoencoder_ensemble=True)
parser.add_argument("--save_dir", required=False, default=None)
parser.add_argument("--ensemble_size", required=False, default=1, type=int)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')


def configure_autoencoder_caller(args):

    if args.compressed_net and not args.unet_skips:
        raise ValueError('Cannot set --compressed_net and alter unet too')

    if not args.compressed_net:
        connector = "with" if args.unet_skips else "without"
        print("Using UNet " + connector + " skips")
        autoencoder_ensemble_caller = default_unet_network if args.unet_skips else no_skip_default_unet_network

    else:
        print('Using compressed (staircase) fully convolutional net')
        autoencoder_ensemble_caller = default_staircase_network

    return autoencoder_ensemble_caller


def set_up_active_learning(args, classification, batch_mode_daf_metric=False, acquisition_kwargs=None):

    # This is for the basic daf case - needs to be parameterised
    encodings_criterion = None
    anchor_criterion = None
    if classification:
        decodings_criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        decodings_criterion = ImageReconstructionLoss(mean_decodings=False, mean_losses=True)

    # Select dataset
    if args.dataset == 'cifar10':
        cifar_dataset = CIFAR10LabelledClassification if classification else CIFAR10Subset
        print('Using cifar10')
    elif args.dataset == 'cifar100':
        cifar_dataset = CIFAR100LabelledClassification if classification else CIFAR100Subset
        print('Using cifar100')

    # Don't want augmentation, just want normalisation
    transform = transform_test
    train_image_dataset = cifar_dataset(
        root='./data', train=True, download=True, transform=transform, 
        original_transform=transform, init_indices=[], target_transform=None
    )

    attr_init = [[None for _ in range(args.ensemble_size)] for _ in range(len(train_image_dataset.data))]
    al_attributes = [
        # Normal initialisation doesn't work because labels size != reconstruction size, so
        # we need to manually input reconstruction shape
        al.dataset_classes.StochasticAttribute('last_logits', attr_init.copy(), args.ensemble_size, cache=False),
        al.dataset_classes.StochasticAttribute('embeddings', attr_init.copy(), args.ensemble_size, cache=False),
    ]
    
    # Standard active learning definitions
    train_dataset = al.dataset_classes.DimensionlessDataset(
        data=torch.tensor(range(len(train_image_dataset.data))),
        labels=torch.tensor(range(len(train_image_dataset.data))),
        costs=torch.tensor([1. for _ in range(len(train_image_dataset.data))]),
        index_class=al.annotation_classes.DimensionlessIndex,
        semi_supervision_agent=None,
        data_reading_method=lambda x: train_image_dataset.get_original_data(x),
        label_reading_method=lambda x: train_image_dataset.get_original_data(x),
        al_attributes=al_attributes,
        # We are using an ensemble in principle
        is_stochastic=True
    )

    round_cost = train_dataset.total_cost * args.minibatch_prop
    total_budget = train_dataset.total_cost * args.total_added_prop

    div_pol = al.batch_querying.NoPolicyBatchQuerying()
    window_class = al.annotation_classes.DimensionlessAnnotationUnit

    # Need this for compatability
    model = ModelWrapper(None)

    if batch_mode_daf_metric:
        metric_function = metric_functions[args.metric_function](train_dataset, **acquisition_kwargs)
        scorer = metric_functions['classification'](train_dataset)
        selector = al.selector.SubsetSelector(
            round_cost=round_cost, 
            acquisition=metric_function,
            scorer = scorer,
            window_class=window_class,
        )
    else:
        metric_function = metric_functions[args.metric_function](train_dataset)
        selector = al.selector.DimensionlessSelector(
            round_cost=round_cost, 
            acquisition=metric_function,
            window_class=window_class,
            diversity_policy=div_pol,
            selection_mode='argmax'
        )

    agent = al.agent.ActiveLearningAgent(
        train_set=train_dataset,
        batch_size=args.batch_size,
        selector_class=selector,
        model=model,
        device=device,
        budget=total_budget
    )

    return agent, train_image_dataset, round_cost, total_budget, encodings_criterion, decodings_criterion, anchor_criterion



if __name__ == '__main__':

    print('CONFIGURING ARGS', flush=True)
    args = parser.parse_args()

    manualSeed = random.randint(1, 10000)
    print('\nSeed:', manualSeed, '\n')
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(manualSeed)

    agent, train_image_dataset, round_cost, total_budget, \
        encodings_criterion, decodings_criterion, anchor_criterion = \
        set_up_active_learning(args, classification=False)
    save_dir = config_savedir(args.save_dir, args)

    print('Adding each round: ', round_cost, ' images')
    print('Will stop at: ', total_budget, ' images')

    model_init_method = configure_autoencoder_caller(args)
    dataloader_init_method = lambda ds, batch_size: DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    agent.init(args.total_start_prop)

    agent = unsupervised_recalibration_script(
        agent, args, model_init_method, dataloader_init_method, train_image_dataset, 
        encodings_criterion, decodings_criterion, anchor_criterion, save_dir, device
    )

    torch.save(agent.model.state_dict(), os.path.join(save_dir, 'final_autoencoder.mdl'))
