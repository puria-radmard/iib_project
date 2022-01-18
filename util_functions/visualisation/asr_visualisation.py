import sys
from tqdm import tqdm
import pandas as pd
import numpy as np

sys.path.insert(0, '../active_learning-pr450')
from alignment_utils import DecodingModelSurrogate
import alignment_tk



SPEAKING_BAND_TRANSLATION = {
    'A1': 'A1',
    'A1_High': 'A1',
    'B2_High': 'B2',
    'B2': 'B2',
    'B1': 'B1',
    'A2_High': 'A2',
    'A2': 'A2',
    'B1_High': 'B1',
    'C1': 'C',
    'C1_High': 'C',
    'pre-A1': 'A1',
    'C2': 'C',
}

TMR_SPEAKING_BAND_TRANSLATION = {
    'A1': 'A',
    'A1_High': 'A',
    'B2_High': 'B2',
    'B2': 'B2',
    'B1': 'B1',
    'A2_High': 'A',
    'A2': 'A',
    'B1_High': 'B1',
    'C1': 'C',
    'C1_High': 'C',
    'pre-A1': 'A',
    'C2': 'C',
}

def group_by_band_groups(band_sep_dict, translation_dict):
    res = {k:0 for k in set(translation_dict.values())}
    for raw_band, num in band_sep_dict.items():
        true_band = translation_dict[raw_band]
        res[true_band] += num
    return res

def turn_dict_into_proportions(dictionary):
    total = sum(dictionary.values())
    return {k: v/total for k, v in dictionary.items()}


def get_band_distribution(utt_list_set, model):

    # This is where we will store number of utterances, and total duration
    # of utterances, in selected_bulats
    band_counts, band_durs = {}, {}

    # For each row (utterance) in the DecodingModelSurrogate engine...
    for i, row in tqdm(model.df.iterrows(), total=len(model.df.index)):

        # If the utterance id is seen in selected_bulats...
        if row.utt in utt_list_set:

            # Add one to the relevant counter
            band_counts[row.SpeakingBand] = band_counts.get(row.SpeakingBand, 0) + 1

            # Add the duration to the relevant timer
            band_durs[row.SpeakingBand] = band_durs.get(row.SpeakingBand, 0) + float(row.dur)

            # Remove that utterance id from the utt list
            utt_list_set.remove(row.utt)

    # Return counts, durations, and the utterances that have not been accounted for
    return band_counts, band_durs, utt_list_set


    

def get_band_distribution_from_path(utt_list_path, model):

    """
        Given a path to a utt list, and a DecodingModelSurrogate instance,
        return the counts and durations of each band, and the utterances
        not accounted for.
    """

    # Open the file path and get all the utterance ids in `selected_bulats'
    with open(utt_list_path, 'r') as f:
        lines = f.read().split('\n')[:-1]
        selected_bulats = set(filter(lambda x: x[0]!='A', lines))

    # Hand over to previous function
    band_counts, band_durs, selected_bulats = get_band_distribution(selected_bulats, model)

    return band_counts, band_durs, selected_bulats

def filter_lines_by_prefix(lines, prefix):
    return list(filter(lambda x: x.startswith(prefix), lines))

def map_lines_by_known_bounds(lines, start_bound, end_bound):
    map_func = lambda x: x.split(start_bound)[1].split(end_bound)[0]
    return list(map(map_func, lines))


def get_information_from_res_file(score_path):
    # Open the .res path and read all the lines
    with open(score_path, 'r') as f:
        all_lines = f.read().split('\n')[:-1]

    # Find all utt ids in the order they appear in the .res path
    utt_lines = filter_lines_by_prefix(all_lines, 'Aligned transcription: ')
    utt_ids = map_lines_by_known_bounds(utt_lines, 'Aligned transcription: ', '.lab')

    # For each of the references, get the word length (-1 for "LAB: ")
    reference_lines = filter_lines_by_prefix(all_lines, ' LAB: ')
    utt_reference_lengths = list(map(lambda x: len(x.split())-1, reference_lines))

    # Get lines with score information on them
    score_lines = filter_lines_by_prefix(all_lines, '    | >')

    # From those, get the code of the part
    score_ids = map_lines_by_known_bounds(score_lines, '| >', ' |')

    # Also from those, get the number of sentences for each part
    snt_nums = map_lines_by_known_bounds(score_lines, ' |    ', '  |')
    # NB: for some reason, sum(snt_nums) might not == len(utt_reference_lengths)

    return utt_ids, utt_reference_lengths, score_ids, score_lines, snt_nums
    

def wers_from_res_file(score_path, metadata_path, lookup_path):
    
    # Get speaker lookups (different code types)
    lookup, _ = alignment_tk.utils.make_lookup_table(lookup_path)
    
    # Get the metadata df and convery speaker code types
    metadata_df = pd.read_csv(metadata_path, delimiter = "\t")
    metadata_df['speaker'] = metadata_df['CustomerToken'].map(lambda x: lookup.get(x, '### NULL'))
    metadata_df = metadata_df[metadata_df['speaker'] != '### NULL']
    
    utt_ids, utt_reference_lengths, score_ids, score_lines, snt_nums = get_information_from_res_file(score_path)

    # Iterate through the utterances, occasionally changing the score/code we are looking at
    score_id_pointer = 0
    utt_id_pointer = 0

    # Make sure we've looked at all utterance ids
    utts_checker = [False for _ in utt_ids]

    while utt_id_pointer < len(utt_ids):
        
        # e.g. 'CBL304-00069-XXXXXXXX-SC0001-en_XXXXXXX_0000000_0006025'
        uid = utt_ids[utt_id_pointer]
        c_code, num_code, _, utt_code, _ = uid.split('-')

        # e.g. '0069SC'
        proposed_score_code = num_code[1:] + utt_code[:2]

        if score_ids[score_id_pointer] != proposed_score_code:
            score_id_pointer += 1
            utt_id_pointer -= 1
        else:
            utts_checker[utt_id_pointer] = True

        utt_id_pointer += 1
    
    print("All utterances checked:", all(utts_checker))
    
    return 