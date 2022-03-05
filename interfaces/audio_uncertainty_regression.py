import os, json, torch
from torch import nn
import numpy as np
from classes_losses import multitask_distillation
from classes_utils.audio.data import AudioUtteranceDataset
from config.ootb import recurrent_regression, las_reg
from interfaces.cifar_acquisition_regression import get_acquisition_prediction_criterion_class, get_unseen_model_output
from training_scripts.cifar_daf_scripts import train_daf_acquisition_regression
from util_functions.data import *
from config import device
import argparse
from util_functions.base import config_savedir
from util_functions.data import coll_fn_utt_multitask

parser = argparse.ArgumentParser()

# Entropies not allowed for speech
viable_secondary_losses = [
    'lc_bce', 'lc_parametric_jacobian_bce', 'lc_histogram_jacobian_bce', 'lc_soft_rank'
]

# These are the allowable multitask cases
# Since we incur massive losses from the lambda scheduling anyway, we don't differentiate between
# the distillation/information rich option and the multitask options
information_rich_losses = ['bag_of_phonemes_multitask', 'phoneme2vec_multitask']


parser.add_argument("--architecture_name", required=True, type=str)
parser.add_argument("--keep_ami", action='store_true')
parser.add_argument("--cell_type", required=False, type=str, default='gru')
parser.add_argument("--lr", required=True, type=float, help="Learning rate of RAE optimizer")
parser.add_argument("--scheduler_epochs", required=True, nargs="+", type=int)
parser.add_argument("--scheduler_proportion", required=True, type=float)
parser.add_argument("--dropout", required=True, type=float)
parser.add_argument("--weight_decay",required=True,type=float)
parser.add_argument("--batch_size", required=True, type=int)
parser.add_argument("--num_epochs", required=True, type=int)
parser.add_argument("--features_paths",required=True,nargs="+",help="List of paths where .ark files are")
parser.add_argument("--alignment_paths",required=True,nargs="+",help="(mapped) CTM file(s) with certainty scores, e.g. ../active_learning-pr450/models/baseline/CTDF1_b50/tdnn-f/decode_LM1-int_b50.unlabelled_b50/score_mbr_10/unlabelled_b50.utt.map.ctm")
parser.add_argument("--max_seq_len",required=True,type=int)
parser.add_argument("--test_prop",required=True,type=float)
parser.add_argument("--save_dir", required=True)
parser.add_argument("--use_dim_means", required=False, default=True)
parser.add_argument("--use_dim_stds", required=False, default=True)

# We use the same names as cifar_acquisition_regression so that we can use get_criterion_class
# but we actually only allow the lc and ranking ones
parser.add_argument(
    "--criterion", required=True, type=str, choices = viable_secondary_losses + information_rich_losses
)
parser.add_argument(
    "--multitask_training_mode", type=str, required=False, default=None,
    choices = viable_secondary_losses + ['None'],
    help="When using multitask mode, the second head has the option to emulate any of these"   
)
parser.add_argument(
    '--multitask_acquisition_target_weight', type=float, required=False, default=None,
    help="e.g.: Loss = (1 - multitask_acquisition_target_weight) * posterior distillation loss + multitask_acquisition_target_weight * acquisition prediction loss, but this depends on our lambda_scheduler_option"
)
parser.add_argument(
    '--lambda_scheduler_option', type=str, required=False, default='no_lambda_scheduling', choices=multitask_distillation.scheduler_functions,
    help="Scheduling we apply to multitask_acquisition_target_weight, which may not even be a weight, remember!"
)



def generate_words2phonemes_dict(phoneme_lookup_path):
    """
        Generates a phoneme lookup dictionary from a path
        Also returns a phoneme vocabulary
    """
    # Initialise the mapper
    words2phonemes = {}

    # Initialise our phoneme vocabulary
    phoneme_vocab = set()

    # Read the phoneme lookup and iterate through lines
    with open(phoneme_lookup_path, 'r') as f:
        phoneme_lookup_lines = phoneme_lookup_path.read()

        # Each line looks like:
        # 'CA_%partial% 'CA_%partial% c;A_B a;P_E
        for line in phoneme_lookup_lines:
            word, word, *phonemes = line.split()
            phonemes = list(phonemes)

            # Add it to the mapper
            words2phonemes[word] = phonemes

            # Add all of themes to our vocabulary
            phoneme_vocab.update(phonemes)

    # Make this indexable
    phoneme_vocab = sorted(list(phoneme_vocab))
    
    return words2phonemes, phoneme_vocab


def generate_utt2phonemesequence_dict(alignment_paths, words2phonemes_dict):
    """
        Picks up a set of alignment paths and a phoneme lookup dict, and generates
        a dictionary that maps utts in that alignment path to the phoneme sequence

            {"utt_id": ['sequence', 'of', 'phonemes']}
    """
    # Initialise the utt_id to phoneme sequence mapper, which we will add to and use shortly
    utt2phonemesequence = {}

    # Iterate through alignment_paths provided
    for alignment_path in alignment_paths:
        
        # Get the alignment lines, each of which looks like:
        # CBL304-00196-XXXXXXXX-SC0001-en_XXXXXXX_0000000_0006141 1 7.05 0.21        AND 0.748625
        # i.e. represents one word in one utt
        with open(alignment_path, 'r') as f:
            alignment_path_lines = f.read()

        # Iterate through these lines
        for line in alignment_path_lines:

            # Get the important information from it
            utt_id, *_, word, _ = line.split()

            # Map the word to the phonemes
            word_phonemes = words2phonemes_dict[word]

            # If we've seen this utterance before, extend the phoneme sequence
            # If it's the first word in the utterance, then initialise it
            current_utt_id_phonemes = utt2phonemesequence.get(utt_id, [])
            current_utt_id_phonemes.extend(word_phonemes)
            utt2phonemesequence[utt_id] = current_utt_id_phonemes

    return utt2phonemesequence


def generate_utt2phoneme_vector_mapper(utt2phonemesequence_mapper, phoneme2vec_model_path):
    """
        Another mapper generator
    """

    # Announce for logging:
    print(
        "Training phoneme2vec model for add_phoneme_information_to_data_dict" if phoneme2vec_model_path == None
        else f"Loading phoneme2vec model from {phoneme2vec_model_path} for add_phoneme_information_to_data_dict"
    )

    # TODO: Generate or load the model
    pass

    # Initialise the mapper
    utt2phoneme_vector = {}

    # TODO: Populate the mapper
    pass

    return utt2phoneme_vector


def add_phoneme_information_to_data_dict(
    data_dict, alignment_paths, phoneme_lookup_path, 
    add_phoneme_sequences, add_bag_of_phonemes, add_mean_phoneme2vec_vector,
    phoneme2vec_model_path=None
    ):
    """
        Picks ups an existing data dictionary, the alignment paths used for it,
        and the word->phoneme lookup filepath, and gives the phoneme sequence of each
        utterance segment

        Also then gives the bag of phonemes

        If data_dict already passed through add_certainties_to_data_dict with the same
        alighnment paths, then there will be no errors wrt finding utt ids

        The `add_XXX' arguments determine what we actually add to the data_dict:
            - add_phoneme_sequences adds a list of lists, each containing the actual phoneme sequence in order
            - add_bag_of_phonemes adds a list of count vectors, e.g. torch.tensor([1, 2, 0, ..., 0, 1, ...])
            - add_mean_phoneme2vec_vector trains a word2vec vector on the utt2phonemesequence dictionary we generate, then uses the average

        Went for this boolean implementation in case we end up having >2 tasks
    """

    # First, get all the words in our vocab in, and map them to their phoneme sequence
    words2phonemes, phoneme_vocab = generate_words2phonemes_dict(phoneme_lookup_path)
    vocab_size = len(phoneme_vocab)

    # Now, get all the utterance ids which appear in the CTM files, and get the sequence
    # of phonemes that appear in them. Even if we don't add this, it is essential
    utt2phonemesequence = generate_utt2phonemesequence_dict(alignment_paths, words2phonemes)

    # If we want to add a list of learned vectors, we have to generate them now
    if add_mean_phoneme2vec_vector:
        utt2phoneme_vector = generate_utt2phoneme_vector_mapper(utt2phonemesequence, phoneme2vec_model_path)
        
    ## Finally, we want a list of lists of phonemes, in the order of the 
    # utterances appearing in the data_dict, which we add to the data_dict

    # Initialise for all cases, only a subset will be used:
    phoneme_sequences_list = [] # the list of sequences we are adding to the data dict
    bag_of_phonemes_list = []   # the bag of phonemes representation we are adding to the data dict
    phoneme_vector_list = []    # the phoneme2vec (averaged) reprepsentations we are adding to the data dict

    # Iterate over each utterance (segment) in the data dict already
    for utt_id in data_dict['utterance_segment_ids']:

        # Get the sequence of phonemes for this utt
        phoneme_sequence = utt2phonemesequence[utt_id]

        if add_phoneme_sequences:
            # Add to our sequence list
            phoneme_sequences_list.append(phoneme_sequence)

        if add_bag_of_phonemes:
            # Generate the bag of phonemes representation
            bag_of_phonemes_representation = torch.zeros(vocab_size)
            for phoneme in phoneme_sequence:
                phoneme_idx = phoneme_vocab.index(phoneme)
                bag_of_phonemes_representation[phoneme_idx] += 1.

            # Turn into logits
            import pdb; pdb.set_trace()
            raise Exception('Turn into logits')
            
            # Add to our bag of phonemes list
            bag_of_phonemes_list.append(bag_of_phonemes_representation)

        if add_mean_phoneme2vec_vector:
            # Add to our mean phoneme2vec vector list
            phoneme_vector_list.append(utt2phoneme_vector[utt_id])

    # Finally, add products to the data_dict and return it
    if add_phoneme_sequences:
        data_dict['phoneme_sequences'] = phoneme_sequences_list
    if add_bag_of_phonemes:
        data_dict['bag_of_phonemes_logits'] = bag_of_phonemes_list
    if add_mean_phoneme2vec_vector:
        data_dict['phoneme2vec_average'] = phoneme_vector_list

    return data_dict


def generate_speech_regression_data_dict(
    features_paths, alignment_paths, max_seq_len, training_mode, second_training_mode, no_ami=True
    ):

    input_keys, target_keys = [], []

    print('started generating data')
    
    # Generate the data dictionary, including the uncertainties
    data_dict = generate_data_dict_utt(features_paths, exclude_ami=no_ami, text_path=None)

    print('generated data_dict')

    # Add the certainties to the data_dict, needed in all cases
    # Make this negative to reflect the LC case (-1 = most certain)
    data_dict = add_certainties_to_data_dict(data_dict, alignment_paths, negative=True)

    print('harvested certainties')

    # Limit the size of each sequence, for CUDA sake
    data_dict = data_dict_length_split(data_dict, max_seq_len)

    print(f'limited mfcc sequence length to {max_seq_len} frames')

    ## Add the non-acquisition information, dependent on the training_mode
    # If the training mode is a multitask one, we need to add more information somehow
    # i.e. the `information_rich` loss that we use later
    if 'bag_of_phonemes' in training_mode:

        # add the phones
        data_dict = add_phoneme_information_to_data_dict(
            data_dict, alignment_paths, 'data/phoneme_lookup.txt',
            add_phoneme_sequences=False, add_bag_of_phonemes=True, 
            add_mean_phoneme2vec_vector=False, phoneme2vec_model_path=None
        )

        # Make sure we pick this one as the first target in the tuple (see) coll_fn_utt_multitask
        target_keys.append('bag_of_phonemes_logits')

    else:
        # Covers all cases here for now
        target_keys.append('certainties')

    # It doesn't actually matter what the secondary training mode is
    # This will always be a confidence (scalar) target => covered by add_certainties_to_data_dict
    # This conditional just makes sure we don't give a second input where not needed
    if second_training_mode != 'None':
        target_keys.append('certainties')

    # Same for all cases right now
    input_keys.append('padded_audio')

    return data_dict, input_keys, target_keys


def sigmoid_muzzle(self, x):
    # Simply pass the unactivated model output through a sigmoid, 
    # making it good for lc losses
    preactivation = self.old_forward_method(x)
    return torch.sigmoid(preactivation)


def generate_speech_multitask_muzzle(self, split_num_dims, split_processes):
    # This generates the function we actually use as our muzzle
    # This function gets an input of size [N,...,sum(split_num_dims)]
    # and outputs a tuple, split by the last dimension with the sizes specified,
    # and with each member of the tuple pass through the corresponding split_processes

    # see also: MultitaskHead
    
    def _muzzle(self, x):

        # Make sure we are dealing with the right size
        assert x.shape[-1] == sum(split_num_dims)

        # Split by the correct sizes
        res = []
        for i in range(len(split_num_dims)):

            # Get the dimensions we are taking for this item of the tuple
            start_idx = sum(split_num_dims[:i])
            end_idx = start_idx + split_num_dims[i]

            # Split by that dimension, without knowing the other dimensiosn
            split_dims = x[...,start_idx:end_idx]

            # Pass through the correct activation, and add to result
            res.append(split_processes[i](split_dims))

        return res

    return _muzzle


def configure_speech_regression_model(architecture_name, dropout, training_mode, data_dict):

    ## Get the muzzle and the muzzle size
    if 'lc' in training_mode:
        # lc is the only thing we have now, so we just do a sigmoid on that
        # Use this for lc_soft_rank as well, for simplicity
        head_size = 1
        muzzle = sigmoid_muzzle

    elif training_mode == 'bag_of_phonemes_multitask':
        # Get the last dimension for each instance and use that like lc,
        # then use the rest as a posterior.
        # Recall that from MultiTaskBagOfPhonemesLoss, we forward inputs directly, as in we
        # return tuples like in the CIFAR case

        # Get the number of phonemes we are bagging, from data_dict
        # data_dict['bag_of_phonemes'] is shape [dataset size, num phonemes]
        information_rich_num_types = data_dict['bag_of_phonemes'].shape[1]

        # number of heads is one greater than the number of phoneme types, i.e. includes the confidence
        head_size = information_rich_num_types + 1

        # muzzle function, requires the way we split each item, and teh way we process each one
        muzzle = generate_speech_multitask_muzzle(
            split_num_dims = (information_rich_num_types, 1),
            split_processes = (torch.nn.Identity, torch.sigmoid)
        )
    
    # Build the model if we are using a recurrent architecture
    if 'recurrent_regression_network' in architecture_name:
        raise NotImplementedError('Not implemented multitask stuff for recurrent_regression_network architectures')
        model_func = getattr(recurrent_regression, architecture_name)
        model = model_func(cell_type, dropout, device)

    # Build the model if we are using a listener (downsampler) + attention network
    elif 'listener' in architecture_name:
        model_func = getattr(las_reg, architecture_name)
        model = model_func(dropout, head_size).to(device)

    ## Add the muzzle we just constructed 
    model.decoder.old_forward_method = model.decoder.forward_method # Preserve the old forward method
    model.decoder.forward_method = muzzle.__get__(model.decoder)    # Add the muzzle

    return model


if __name__ == '__main__':

    args = parser.parse_args()

    # Get the data dictionary that we will torchify shortly
    # This will also give us the attributes that are used for input and output
    # The input value, for now, is always 
    data_dict, input_keys, target_keys = generate_speech_regression_data_dict(
        args.features_paths, args.alignment_paths, args.max_seq_len, 
        args.criterion, args.multitask_training_mode, not args.keep_ami
    )

    model = configure_speech_regression_model(args.architecture_name, args.dropout, args.criterion, data_dict)

    # Training objects
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.scheduler_proportion)

    # Compile the main dataset that contains everything
    master_dataset = AudioUtteranceDataset(
        data_dict["mfcc"], data_dict["utterance_segment_ids"],
        "config/per_speaker_mean.pkl", "config/per_speaker_std.pkl",
        confidence = data_dict["certainties"]
    )

    # Split the data into train and test
    test_length = int(np.floor(args.test_prop*len(master_dataset)))
    train_length = len(master_dataset) - test_length
    datasettrn, datasettst = torch.utils.data.random_split(master_dataset, [train_length, test_length])

    # This is our collation function for the dataloaders, the form of which is documented in the
    # coll_fn_utt_multitask docstring
    collation_function = coll_fn_utt_multitask(input_keys, target_keys)

    # Make the dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        datasettrn, collate_fn=collation_function, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        datasettst, collate_fn=collation_function, batch_size=args.batch_size, shuffle=True
    )

    print('DONE CONFIGURING DATA\n', flush=True)

    save_dir = config_savedir(args.save_dir, args)

    # Get the criterion class - lifted directly from cifar_acquisition_regression
    criterion = get_acquisition_prediction_criterion_class(
        training_mode=args.criterion,
        # Should be safe since we are not using entropy anywhere
        dataset=None,
        objective_uniformalisation='no_un',
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        second_training_mode=args.multitask_training_mode,
        args=args,
        save_dir=save_dir,
        lambda_scheduling=args.lambda_scheduler_option
    )

    import pdb; pdb.set_trace()

    # Given the new dataloader form, this script will be better suited

    model, results = train_daf_acquisition_regression(
        regressor=model,
        optimizer=opt,
        scheduler=scheduler,
        scheduler_epochs=args.scheduler_epochs,
        decodings_criterion=criterion,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        num_epochs=args.num_epochs,
        show_print=True
    )

    with open(os.path.join(save_dir, 'results.json'), 'w') as jf:
        json.dump(results, jf)

    # The results we need
    # TODO: make this for train_dataset too
    all_acquisition, all_preds = get_unseen_model_output(datasettst, args.batch_size, model)

    torch.save(
        {'all_acquisition': all_acquisition, 'all_preds': all_preds}, 
        os.path.join(save_dir, 'acquisition_results.pkl')
    )

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.mdl'))
    torch.save(criterion, os.path.join(save_dir, 'criterion.pkl'))
