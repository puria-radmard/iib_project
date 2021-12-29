import os, json, argparse, torch, pickle
import numpy as np
from re import S
import random
from torch import nn

from cifar_repo.utils.logger import Logger
from cifar_repo.cifar import transform_test

import active_learning as al
from config.ootb_architectures.library import default_audio_unet_network, default_noskip_audio_unet_network

from training_scripts.audio_training_scripts import train_autoencoder_ensemble
from classes_losses.reconstrution import ReconstructionLoss
from classes_utils.audio.data import SubsetAudioUtteranceDataset
from classes_utils.architecture import SkipEncoderDecoderEnsemble
from classes_architectures.cifar.encoder import DEFAULT_UNET_ENCODER_KERNEL_SIZES, DEFAULT_UNET_ENCODER_STRIDES, DEFAULT_UNET_ENCODER_OUT_CHANNELS
from classes_architectures.cifar.decoder import DEFAULT_UNET_DECODER_OUT_CHANNELS, DEFAULT_UNET_DECODER_KERNEL_SIZES, DEFAULT_UNET_DECODER_STRIDES, DEFAULT_UNET_DECODER_CONCATS
from training_scripts.unsupervised_recalibration_scripts import unsupervised_recalibration_script
from util_functions.data import *
from config import metric_functions
from torch.utils.data import DataLoader
from interfaces.cifar_unsupervised_recalibration import ModelWrapper, config_savedir

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

    model_init_method = default_audio_unet_network if args.unet_skips else default_noskip_audio_unet_network
    dataloader_init_method = lambda ds, bs: DataLoader(
        ds, collate_fn=coll_fn_utt_with_channel_insersion, batch_size=bs, shuffle=True
    )

    agent = unsupervised_recalibration_script(
        agent, args, model_init_method, dataloader_init_method, train_audio_dataset, 
        encodings_criterion, decodings_criterion, anchor_criterion, save_dir, device
    )

    torch.save(agent.model.state_dict(), os.path.join(save_dir, 'final_autoencoder.mdl'))
