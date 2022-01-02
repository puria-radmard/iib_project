import os, json, argparse, torch
import random
from torch import nn

from cifar_repo.cifar import transform_test

import active_learning as al

from interfaces.cifar_unsupervised_recalibration import set_up_active_learning
from classes_losses.reconstrution import ImageReconstructionLoss
from classes_utils.cifar.data import CIFAR100Subset, CIFAR10Subset
from config.ootb_architectures import default_staircase_network, default_unet_network, no_skip_default_unet_network
from training_scripts.unsupervised_recalibration_scripts import unsupervised_recalibration_script
from util_functions.data import *
from config import metric_functions
from torch.utils.data import DataLoader
from util_functions.base import config_savedir



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
parser.set_defaults(reinitialise_autoencoder_ensemble=False)
parser.add_argument("--save_dir", required=False, default=None)
parser.add_argument("--ensemble_size", required=False, default=1, type=int)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')



def labelled_classification_losses(args):
    return encodings_criterion, decodings_criterion, anchor_criterion


def configure_labelled_classifier_caller(args):
    return labelled_classifier_caller



if __name__ == '__main__':

    print('CONFIGURING ARGS', flush=True)
    args = parser.parse_args()

    manualSeed = random.randint(1, 10000)
    print('\nSeed:', manualSeed, '\n')
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(manualSeed)

    agent, train_image_dataset, round_cost, total_budget, *_ = set_up_active_learning(args)
    encodings_criterion, decodings_criterion, anchor_criterion = labelled_classification_losses(args)
    save_dir = config_savedir(args.save_dir, args)

    print('Adding each round: ', round_cost, ' images')
    print('Will stop at: ', total_budget, ' images')

    model_init_method = configure_labelled_classifier_caller(args)
    dataloader_init_method = lambda ds, bs: DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4)

    agent.init(args.total_start_prop)

    agent = unsupervised_recalibration_script(
        agent, args, model_init_method, dataloader_init_method, train_image_dataset, 
        encodings_criterion, decodings_criterion, anchor_criterion, save_dir, device
    )

    torch.save(agent.model.state_dict(), os.path.join(save_dir, 'final_autoencoder.mdl'))
