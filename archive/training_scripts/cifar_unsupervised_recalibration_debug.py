import os, json, argparse, torch, pickle
import random
from torch import nn

from cifar_repo.cifar import transform_test

import active_learning as al

from classes_losses.reconstrution import ImageReconstructionLoss
from classes_utils.cifar.data import CIFAR100Subset, CIFAR10Subset
from classes_utils.architecture import SkipEncoderDecoderEnsemble, EncoderDecoderEnsemble
from classes_architectures.cifar.encoder import DEFAULT_UNET_ENCODER_KERNEL_SIZES, DEFAULT_UNET_ENCODER_STRIDES, DEFAULT_UNET_ENCODER_OUT_CHANNELS
from classes_architectures.cifar.decoder import DEFAULT_UNET_DECODER_OUT_CHANNELS, DEFAULT_UNET_DECODER_KERNEL_SIZES, DEFAULT_UNET_DECODER_STRIDES, DEFAULT_UNET_DECODER_CONCATS
from training_scripts.cifar_daf_scripts import train_autoencoder_ensemble
from util_functions.data import *
from config import metric_functions

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


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


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.midloop = False
    def forward(self, x, *args, **kwargs):
        encodings, decodings = self.model(x)
        if self.midloop:
            # Training, so return logits
            return encodings, decodings
        else:
            return {'last_logits': decodings, 'embeddings': encodings}


def update_indices(previously_trained_indices, agent):
    # We are using the pytorch dataloader, so need to move the batches to the CIFARSubset classes
    new_indices = []
    for ls in agent.labelled_set:
        # The agent will continually add more data to the labelled set
        # We want to filter out the previously seen indices, only finetuning on new data
        unseen_indices = list(filter(lambda x: x not in previously_trained_indices, ls))
        new_indices.extend(unseen_indices)
        previously_trained_indices.update(unseen_indices)  
    return previously_trained_indices, new_indices


def configure_autoencoder(args):

    if args.compressed_net and not args.unet_skips:
        raise ValueError('Cannot set --compressed_net and alter unet too')

    if not args.compressed_net:

        connector = "with" if args.unet_skips else "without"
        print("Using UNet " + connector + " skips")

        encoder_ensemble_kwargs = {
            "input_size": (3, 32, 32),
            "out_channels": DEFAULT_UNET_ENCODER_OUT_CHANNELS,
            "kernel_sizes": DEFAULT_UNET_ENCODER_KERNEL_SIZES,
            "strides": DEFAULT_UNET_ENCODER_STRIDES, 
            "variational": False
        }

        concat_idxs = DEFAULT_UNET_DECODER_CONCATS if args.unet_skips else [-1 for _ in DEFAULT_UNET_DECODER_CONCATS]

        decoder_ensemble_kwargs = {
            "embedding_dim": 256,
            "output_channels": 3,
            "out_channels": DEFAULT_UNET_DECODER_OUT_CHANNELS,
            "kernel_sizes": DEFAULT_UNET_DECODER_KERNEL_SIZES,
            "strides": DEFAULT_UNET_DECODER_STRIDES, 
            "concat_idxs": concat_idxs,
        }

        autoencoder_ensemble = SkipEncoderDecoderEnsemble(
            ensemble_type='basic',
            encoder_type='unet',
            decoder_type='unet',
            ensemble_size=args.ensemble_size,
            encoder_ensemble_kwargs=encoder_ensemble_kwargs,
            decoder_ensemble_kwargs=decoder_ensemble_kwargs,
            mult_noise=0
        )

    else:

        print('Using compressed fully convolutional net')


        encoder_ensemble_kwargs = {
            "input_size": (3, 32, 32),
            "fc_sizes": [],
            "out_channels": [16, 24, 128],
            "kernel_sizes": [3, 3, 3],
            "strides": [2, 2, 2],
            "variational": False,
            "base_padding": 1,
            "sequence_padding": False
        }

        decoder_ensemble_kwargs = {
            "channels": [128, 32, 32, 16, 16, 3, 3],
            "kernels": [3, 2, 3, 2, 3, 2],
            "strides": [2, 1, 2, 1, 2, 1],
            "paddings": [0, 0, 0, 0, 0, 0],
            "sequence": "DCDCDC"
        }

        autoencoder_ensemble = EncoderDecoderEnsemble(
            ensemble_type='basic',
            encoder_type='no_skip',
            decoder_type='staircase',
            ensemble_size=1,
            encoder_ensemble_kwargs=encoder_ensemble_kwargs,
            decoder_ensemble_kwargs=decoder_ensemble_kwargs,
            mult_noise=0
        )

    return autoencoder_ensemble


def config_savedir(args):
  
    i=0
    while True:
        try:
            save_dir=f"{args.save_dir}-{i}"
            os.mkdir(save_dir)
            break
        except:
            i+=1
        if i>50:
            raise Exception("Too many folders!")

    saveable_args = vars(args)
    config_json_path = os.path.join(save_dir, "config.json")

    print(f"Config dir : {save_dir}\n", flush=True)

    with open(config_json_path, "w") as jfile:
        json.dump(saveable_args, jfile)

    return save_dir


def set_up_active_learning(args):

    # This is for the basic daf case - needs to be parameterised
    encodings_criterion = None
    decodings_criterion = ImageReconstructionLoss(mean_decodings=False, mean_losses=True)
    anchor_criterion = None

    # Select dataset
    if args.dataset == 'cifar10':
        cifar_dataset = CIFAR10Subset
        print('Using cifar10')
    elif args.dataset == 'cifar100':
        cifar_dataset = CIFAR100Subset
        print('Using cifar100')

    # Don't want augmentation, just want normalisation
    transform = transform_test
    train_image_dataset = cifar_dataset(
        root='./data', train=True, download=True, transform=transform, 
        original_transform=transform, init_indices=[], target_transform=None
    )

    # Standard active learning definitions
    train_dataset = al.dataset_classes.DimensionlessDataset(
        data=torch.tensor(range(len(train_image_dataset.data))),
        labels=torch.tensor(range(len(train_image_dataset.data))),
        costs=torch.tensor([1. for _ in range(len(train_image_dataset.data))]),
        index_class=al.annotation_classes.DimensionlessIndex,
        semi_supervision_agent=None,
        data_reading_method=lambda x: train_image_dataset.get_original_data(x),
        label_reading_method=lambda x: train_image_dataset.get_original_data(x),
        al_attributes=[
            # Normal initialisation doesn't work because labels size != reconstruction size, so
            # we need to manually input reconstruction shape
            al.dataset_classes.StochasticAttribute(
                'last_logits', [[None for _ in range(args.ensemble_size)] for _ in range(len(train_image_dataset.data))], 
                args.ensemble_size, cache=False
            ),
            al.dataset_classes.StochasticAttribute(
                'embeddings', [[None for _ in range(args.ensemble_size)] for _ in range(len(train_image_dataset.data))], 
                args.ensemble_size, cache=False
            ),
        ],
        # We are using an ensemble in principle
        is_stochastic=True
    )

    round_cost = train_dataset.total_cost * args.minibatch_prop
    total_budget = train_dataset.total_cost * args.total_added_prop

    metric_function = metric_functions[args.metric_function](train_dataset)

    div_pol = al.batch_querying.NoPolicyBatchQuerying()
    window_class = al.annotation_classes.DimensionlessAnnotationUnit
    selector = al.selector.DimensionlessSelector(
        round_cost=round_cost, 
        acquisition=metric_function,
        window_class=window_class,
        diversity_policy=div_pol,
        selection_mode='argmax'
    )

    # Need this for compatability
    model = ModelWrapper(None)

    agent = al.agent.ActiveLearningAgent(
        train_set=train_dataset,
        batch_size=args.batch_size,
        selector_class=selector,
        model=model,
        device=device,
        budget=total_budget
    )

    return agent, train_image_dataset, round_cost, total_budget, encodings_criterion, decodings_criterion, anchor_criterion


def make_metric_dictionary(all_round_windows, round_num, labelled_indices, newly_added_indices, save_dir):
    
    # Initialise the distributions
    labelled_distribution, unlabelled_distribution, new_added_distribution = [], [], []

    # Selector stores all window scores anyway, which are what the `active learning' is based on
    for window in all_round_windows:
        # Get score
        metric, index = window.score, window.i
        # Add to correct distributions
        is_labelled = index in labelled_indices
        labelled_distribution.append(metric) if is_labelled else unlabelled_distribution.append(metric)
        # Another distribution of indices about to be labelled
        if index in newly_added_indices:
            new_added_distribution.append(metric)
    
    if round_num%3 == 1:
        fig_path = os.path.join(save_dir, f"dis_plot_{round_num}_{len(labelled_distribution)}.png")
        fig, axs = plt.subplots(1, figsize=(10, 10))
        sns.histplot(unlabelled_distribution, label = "Unseen set", ax=axs, color='r')
        sns.histplot(labelled_distribution, label = "Seen set", ax=axs, color='b')
        sns.histplot(new_added_distribution, label = "New added set", ax=axs, color='g')
        axs.legend()
        fig.savefig(fig_path)

    pickle_path = os.path.join(save_dir, f"distributions_{round_num}_{len(labelled_distribution)}.pkl")
    with open(pickle_path, 'wb') as h:
        pickle.dump(
            {'seen': labelled_distribution, 'unseen': unlabelled_distribution},
            h, pickle.HIGHEST_PROTOCOL
        )


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
        set_up_active_learning(args)
    save_dir = config_savedir(args)

    print('Adding each round: ', round_cost, ' images')
    print('Will stop at: ', total_budget, ' images')

    agent.init(args.total_start_prop)

    round_num = 0
    # Keep track of the images we have already trained on
    # At each finetuning stage we finetune only on the added set
    previously_trained_indices = set()

    for _ in agent:

        # Lag in when the metric values are actually updated!
        if round_num > 0:
            make_metric_dictionary(agent.selector.all_round_windows, round_num, previously_trained_indices, set(new_indices), save_dir)

            import pdb; pdb.set_trace()

        previously_trained_indices, new_indices = update_indices(previously_trained_indices, agent)
        
        ## Sanity check on this
        # unlabelled_scores = list(filter(lambda x: x.i not in previously_trained_indices, agent.selector.all_round_windows))
        # unlabelled_scores = sorted(unlabelled_scores, key=lambda w: w.score, reverse = True)
        # agent_selected_indices = set(map(lambda x: x.i, agent.selector.round_selection))

        # labelled_indices = []
        # for x in agent.labelled_set: labelled_indices.extend(x)
        # set(labelled_indices) == previously_trained_indices

        if args.reinitialise_autoencoder_ensemble:
            # Reinit - we finetune on all data so far
            train_image_dataset.indices = list(previously_trained_indices)
        else:
            # Don't reinit - we only finetune on unseen data
            train_image_dataset.indices = list(new_indices)
        train_dataloader = torch.utils.data.DataLoader(train_image_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # Make logging path based on save_dir
        round_num += 1

        with open(os.path.join(save_dir, f'labelled_set_{round_num}.txt'), 'w') as f:
            for i in new_indices:
                f.write(str(i))
                f.write('\n')

        print(f'\n\nRound: {round_num} | {len(train_image_dataset)} labelled')

        if args.reinitialise_autoencoder_ensemble or round_num == 1:
            agent.model.model = configure_autoencoder(args)
            agent.model = agent.model.to(device)

        # Need to return logits drectly while in training/val script
        agent.model.midloop = True

        if round_num == 1:
            initial_optimizer = torch.optim.SGD(agent.model.parameters(), lr=args.initial_lr, \
                momentum=args.momentum, weight_decay=args.weight_decay)
            agent.model, results = train_autoencoder_ensemble(
                ensemble=agent.model,
                optimizer=initial_optimizer,
                scheduler=None,
                scheduler_epochs=[],
                encodings_criterion=encodings_criterion,
                decodings_criterion=decodings_criterion,
                anchor_criterion=anchor_criterion,
                train_dataloader=train_dataloader,
                test_dataloader=[],
                num_epochs=args.num_initial_epochs,
            )

        else:
            finetune_optimizer = torch.optim.SGD(agent.model.parameters(), lr=args.finetune_lr, \
                momentum=args.momentum, weight_decay=args.finetune_weight_decay)
            agent.model, results = train_autoencoder_ensemble(
                ensemble=agent.model,
                optimizer=finetune_optimizer,
                scheduler=None,
                scheduler_epochs=[],
                encodings_criterion=encodings_criterion,
                decodings_criterion=decodings_criterion,
                anchor_criterion=anchor_criterion,
                train_dataloader=train_dataloader,
                test_dataloader=[],
                num_epochs=args.num_finetune_epochs,
            )
            
        results_path = os.path.join(save_dir, f'round_{round_num}_results.json')
        with open(results_path, 'w') as jfile:
            json.dump(results, jfile)

        # Need to return attribute dictionary while for active learning agent
        agent.model.midloop = False

        # When this loop ends, model attributes and scores are updated, using a model that is trained on everything so far
        # So we need to visualise the distributions after this point

    torch.save(agent.model.state_dict(), os.path.join(save_dir, 'final_autoencoder.mdl'))
