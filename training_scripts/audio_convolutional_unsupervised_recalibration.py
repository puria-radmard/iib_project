import os, json, argparse, torch, pickle
import numpy as np
from re import S
import random
from torch import nn

from cifar_repo.utils.logger import Logger
from cifar_repo.cifar import transform_test

import active_learning as al

from training_scripts.audio_training_scripts import train_autoencoder_ensemble
from classes_losses.reconstrution import ReconstructionLoss
from classes_utils.audio.data import SubsetAudioUtteranceDataset
from classes_utils.architecture import SkipEncoderDecoderEnsemble
from classes_architectures.cifar.encoder import DEFAULT_UNET_ENCODER_KERNEL_SIZES, DEFAULT_UNET_ENCODER_STRIDES, DEFAULT_UNET_ENCODER_OUT_CHANNELS
from classes_architectures.cifar.decoder import DEFAULT_UNET_DECODER_OUT_CHANNELS, DEFAULT_UNET_DECODER_KERNEL_SIZES, DEFAULT_UNET_DECODER_STRIDES, DEFAULT_UNET_DECODER_CONCATS
from util_functions.data import *
from config import metric_functions

import matplotlib.pyplot as plt
import seaborn as sns

from util_functions.data import coll_fn_utt_with_channel_insersion
sns.set_style('darkgrid')


device = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu')
)
use_cuda = torch.cuda.is_available()

al.disable_tqdm()


parser = argparse.ArgumentParser()
parser.add_argument("--metric_function", required=True, type=str)
parser.add_argument('--unet_with_skips', dest='unet_skips', action='store_true')
parser.add_argument('--unet_without_skips', dest='unet_skips', action='store_false')
parser.set_defaults(unet_skips=True)
parser.add_argument('--data_aug_type', required=False, type=str, default="none")
parser.add_argument("--minibatch_seconds", required=True, type=int)
parser.add_argument("--total_added_seconds", required=True, type=int)
parser.add_argument("--initial_lr", required=True, type=float, help="Learning rate of RAE optimizer")
parser.add_argument("--finetune_lr", required=True, type=float, help="Learning rate of RAE optimizer")
parser.add_argument("--scheduler_epochs", required=True, nargs="+", type=int)
parser.add_argument("--scheduler_proportion", required=True, type=float)
parser.add_argument("--dropout", required=True, type=float)
parser.add_argument("--weight_decay",required=True,type=float)
parser.add_argument("--finetune_weight_decay",required=True,type=float)
parser.add_argument("--batch_size", required=True, type=int)
parser.add_argument("--num_initial_epochs", required=True, type=int)
parser.add_argument("--num_finetune_epochs", required=True, type=int)
parser.add_argument("--labelled_utt_list_path", required=True, type=str)
parser.add_argument("--unlabelled_utt_list_path", required=True, type=str)
parser.add_argument("--data_dict_path",required=False,help="Pre-prepared data_dict, overrides all other data arguments")
parser.add_argument("--features_paths",required=False,nargs="+",help="List of paths where .ark files are")
parser.add_argument('--do_reinitialise_autoencoder_ensemble', dest='reinitialise_autoencoder_ensemble', action='store_true')
parser.add_argument('--do_not_reinitialise_autoencoder_ensemble', dest='reinitialise_autoencoder_ensemble', action='store_false')
parser.set_defaults(reinitialise_autoencoder_ensemble=False)
parser.add_argument("--max_seq_len",required=True,type=int)
parser.add_argument("--save_dir", required=False, default=None)
parser.add_argument("--ensemble_size", required=False, default=1, type=int)
parser.add_argument("--use_dim_means", required=False, default=True)
parser.add_argument("--use_dim_stds", required=False, default=True)
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

    encoder_ensemble_kwargs = {
        "input_size": (1, None, 40),
        "out_channels": DEFAULT_UNET_ENCODER_OUT_CHANNELS[:-1],
        "kernel_sizes": DEFAULT_UNET_ENCODER_KERNEL_SIZES[:-1],
        "strides": DEFAULT_UNET_ENCODER_STRIDES[:-1],
        "variational": False,
        "flatten": False
    }

    concat_idxs = [-1, 2, 1] if args.unet_skips else [-1, -1, -1]

    decoder_ensemble_kwargs = {
        "embedding_dim": 256, #None,
        "output_channels": 1,
        "out_channels": DEFAULT_UNET_DECODER_OUT_CHANNELS[1:], #[128, 64, 32],
        "kernel_sizes": DEFAULT_UNET_DECODER_KERNEL_SIZES[1:], #[(4, 3), 3, 3, 4],
        "strides": DEFAULT_UNET_DECODER_STRIDES[1:], #[2, 2, 2, 1], 
        "concat_idxs": concat_idxs,
        "flattened": False
    }

    autoencoder_ensemble = SkipEncoderDecoderEnsemble(
        ensemble_type='basic',
        encoder_type='unet',
        decoder_type='unet',
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
    decodings_criterion = ReconstructionLoss(mean_decodings=False, mean_losses=True)
    anchor_criterion = None

    if args.features_paths:
        print('loading from features path')
        data_dict = generate_data_dict_utt(args.features_paths, text_path=None)
    #     if not args.use_dim_means:
    #         print('removing mean')
    #         data_dict['mfcc'] = data_demeaning(data_dict['mfcc'])
    # elif args.data_dict_path:
    #     print('loading from data dict path')
    #     with open(args.data_dict_path, 'rb') as handle:
    #         data_dict = pickle.load(handle)
    # else:
    #     raise argparse.ArgumentError(None, 'Need either data_dict_path or features_paths')
    
    print(f'limiting mfcc sequence length to {args.max_seq_len}')
    data_dict = data_dict_length_split(data_dict, args.max_seq_len)

    print(f'splitting data into labelled and unlabelled')
    labelled_data_dict, unlabelled_data_dict = split_data_dict(
        data_dict, args.labelled_utt_list_path, args.unlabelled_utt_list_path
    )

    train_audio_dataset = SubsetAudioUtteranceDataset(
        labelled_data_dict["mfcc"] + unlabelled_data_dict["mfcc"],
        labelled_data_dict["utterance_segment_ids"] + unlabelled_data_dict["utterance_segment_ids"],
        labelled_data_dict["text"] + unlabelled_data_dict["text"],
        "config/per_speaker_mean.pkl",
        "config/per_speaker_std.pkl",
        list(range(len(labelled_data_dict["mfcc"])))       
    )

    # Standard active learning definitions
    train_dataset = al.dataset_classes.AudioReconstructionDimensionlessDataset(
        data=torch.tensor(range(len(train_audio_dataset.audio))),
        labels=torch.tensor(range(len(train_audio_dataset.audio))),
        costs=torch.tensor([features.shape[0]*0.01 for features in train_audio_dataset.audio]),
        index_class=al.annotation_classes.DimensionlessIndex,
        semi_supervision_agent=None,
        data_reading_method=lambda x: np.expand_dims(train_audio_dataset.get_original_data(x)[0], 0),
        label_reading_method=lambda x: np.expand_dims(train_audio_dataset.get_original_data(x)[0], 0),
        al_attributes=[
            # Normal initialisation doesn't work because labels size != reconstruction size, so
            # we need to manually input reconstruction shape
            al.dataset_classes.StochasticAttribute(
                'last_logits', [[None for _ in range(args.ensemble_size)] for _ in range(len(train_audio_dataset.audio))], 
                args.ensemble_size, cache=False
            ),
            al.dataset_classes.StochasticAttribute(
                'embeddings', [[None for _ in range(args.ensemble_size)] for _ in range(len(train_audio_dataset.audio))], 
                args.ensemble_size, cache=False
            ),
        ],
        # We are using an ensemble in principle
        is_stochastic=True
    )

    round_cost = args.minibatch_seconds
    total_budget = args.total_added_seconds

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

    return agent, train_audio_dataset, round_cost, total_budget, encodings_criterion, decodings_criterion, anchor_criterion


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

    agent, train_audio_dataset, round_cost, total_budget, \
        encodings_criterion, decodings_criterion, anchor_criterion = \
        set_up_active_learning(args)
    save_dir = config_savedir(args)

    print('Adding each round: ', round_cost, ' seconds')
    print('Will stop at: ', total_budget, ' seconds')

    # Init with range(len(train_audio_dataset))

    # agent.model.model = configure_autoencoder(args)
    # batch = torch.randn(8, 1, 300, 40)
    # agent.model(batch)
    # agent.model.model.decoder(x, skip_list)

    # Starting with 
    budget_spent = 0
    for i in train_audio_dataset.indices:
        agent.train_set.index.label_instance(i)
        budget_spent += agent.train_set.get_cost_by_index(i)
    # 'Global' cost budget variable
    agent.budget -= budget_spent
    agent.update_datasets()
    print("Finished agent index initialisation")
    # Manually resetting agent budget
    agent.budget = total_budget

    round_num = 0
    # Keep track of the utterances we have already trained on
    # At each finetuning stage we finetune only on the added set
    previously_trained_indices = set()

    for _ in agent:

        # Lag in when the metric values are actually updated!
        if round_num > 0:
            make_metric_dictionary(agent.selector.all_round_windows, round_num, previously_trained_indices, set(new_indices), save_dir)

        previously_trained_indices, new_indices = update_indices(previously_trained_indices, agent)
        
        ## Sanity check on this
        # unlabelled_scores = list(filter(lambda x: x.i not in previously_trained_indices, agent.selector.all_round_windows))
        # unlabelled_scores = sorted(unlabelled_scores, key=lambda w: w.score, reverse = True)
        # set(map(lambda x: x.i, unlabelled_scores[:500])) == set(new_indices)

        if args.reinitialise_autoencoder_ensemble:
            # Reinit - we finetune on all data so far
            train_audio_dataset.indices = list(previously_trained_indices)
        else:
            # Don't reinit - we only finetune on unseen data
            train_audio_dataset.indices = list(new_indices)
        train_dataloader = torch.utils.data.DataLoader(train_audio_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        train_dataloader = torch.utils.data.DataLoader(
            train_audio_dataset, collate_fn=coll_fn_utt_with_channel_insersion, batch_size=args.batch_size, shuffle=True
        )
        
        # Make logging path based on save_dir
        round_num += 1

        with open(os.path.join(save_dir, f'labelled_set_{round_num}.txt'), 'w') as f:
            for i in new_indices:
                f.write(str(i))
                f.write('\n')

        print(f'\n\nRound: {round_num} | {len(train_audio_dataset)} labelled')

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
                show_print=True,
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
                show_print=True,
            )

        results_path = os.path.join(save_dir, f'round_{round_num}_results.json')
        with open(results_path, 'w') as jfile:
            json.dump(results, jfile)

        # Need to return attribute dictionary while for active learning agent
        agent.model.midloop = False

        # When this loop ends, model attributes and scores are updated, using a model that is trained on everything so far
        # So we need to visualise the distributions after this point

    torch.save(agent.model.state_dict(), os.path.join(save_dir, 'final_autoencoder.mdl'))