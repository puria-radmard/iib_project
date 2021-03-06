import os, random, argparse, torch

from cifar_repo.cifar import transform_test

from classes_utils.base.data import ClassificationDAFDataloader
from config.ootb import convolutional_classification

from interfaces.cifar_unsupervised_recalibration import set_up_active_learning
from training_scripts.recalibration_scripts import labelled_classification_recalibration_script
from util_functions.data import *
from util_functions.base import config_savedir


device = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu')
)
use_cuda = torch.cuda.is_available()

# al.disable_tqdm()


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--architecture_name", required=True, type=str)
parser.add_argument("--metric_function", required=False, type=str, choices=['classification_with_graph_cut'], default='classification_with_graph_cut')
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
parser.add_argument('--data_aug_type', required=False, type=str, default="none")
parser.add_argument("--batch_size", required=True, type=int)
parser.add_argument("--num_initial_epochs", required=True, type=int)
parser.add_argument("--num_finetune_epochs", required=True, type=int)
parser.add_argument('--do_reinitialise_autoencoder_ensemble', dest='reinitialise_autoencoder_ensemble', action='store_true')
parser.add_argument('--do_not_reinitialise_autoencoder_ensemble', dest='reinitialise_autoencoder_ensemble', action='store_false')
parser.set_defaults(reinitialise_autoencoder_ensemble=True)
parser.add_argument("--save_dir", required=True, default=None)
parser.add_argument("--ensemble_size", required=False, default=1, type=int)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', required=False)

#raise Exception('incorporate acq func names & weightings here and in loop')
#raise Exception('allow option to keep diversity static')


def configure_labelled_classifier_caller(args):
    model_caller = getattr(convolutional_classification, args.architecture_name)
    labelled_classifier_caller = lambda : model_caller(args.dropout, use_logits = True)
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

    size_N = 50000
    size_A = size_N * args.minibatch_prop

    acquisition_kwargs = dict(
        vertex_location_name='embeddings',
        span_importance_weight=1 / (0.5 * size_A * (size_A - 1)),
        vertex_importance_weight=1 / (size_A),
        seperation_importance_weight=1 / (size_A * (size_N - size_A))
    )

    agent, train_image_dataset, round_cost, total_budget, encodings_criterion, decodings_criterion, anchor_criterion = \
        set_up_active_learning(args, classification=True, batch_mode_daf_metric=True, acquisition_kwargs=acquisition_kwargs)
    save_dir = config_savedir(args.save_dir, args)

    print('Adding each round: ', round_cost, ' images')
    print('Will stop at: ', total_budget, ' images')

    model_init_method = configure_labelled_classifier_caller(args)
    dataloader_init_method = lambda ds, batch_size: ClassificationDAFDataloader(ds, collate_fn=None, batch_size=batch_size, num_workers=4)

    agent.init(args.total_start_prop)

    agent = labelled_classification_recalibration_script(
        agent, args, model_init_method, dataloader_init_method, train_image_dataset, 
        encodings_criterion, decodings_criterion, anchor_criterion, save_dir, device,
        make_graphics=False
    )

    torch.save(agent.model.state_dict(), os.path.join(save_dir, 'final_autoencoder.mdl'))
