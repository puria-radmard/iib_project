import pandas as pd
import numpy as np
from kaldiio import load_ark
from tqdm import tqdm
import torch
from torch import nn

__all__ = [
    'chunks',
    'data_range_normalisation',
    'data_demeaning',
    'coll_fn_utt_with_targets',
    'data_dict_length_split',
    'include_utterance',
    'features_from_time_bounds',
    'filter_alignment_df',
    'add_certainties_to_data_dict',
    'generate_data_features',
    'generate_word_indices',
    'generate_data_dict_words',
    'generate_data_dict_utt',
    'split_data_dict',
    'coll_fn_words',
    'coll_fn_utt',
    'generate_coll_fn_simclr_utt',
]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def data_range_normalisation(_features):
    raise Exception('Need new implementation')
    _features = [h for h in _features]
    dim_means = np.concatenate(_features).mean(0)
    _features = [f - dim_means for f in _features]
    dim_maxs = abs(np.concatenate(_features)).max(0)
    _features = [f / dim_maxs for f in _features]
    return _features


def data_demeaning(_features, dim_means=None):
    # np.mean does not work for large mag arrays, so we must divide first then add
    arr = np.concatenate(_features)
    lengths = [len(a) for a in _features]
    N = arr.shape[0]

    if dim_means is None:
        dived_arr = arr / N
        dim_means = dived_arr.sum(0)
    demeaned_arr = arr - dim_means

    out_features = []
    start_idx = 0
    for l in lengths:
        out_features.append(demeaned_arr[start_idx:start_idx+l])
        start_idx += l
    
    return out_features


def data_devariance(_features, dim_means):

    arr = np.concatenate(_features)
    lengths = [len(a) for a in _features]
    N = arr.shape[0]

    total_variance = np.zeros(arr.shape[-1])
    for i in tqdm(range(0, N, 100)):
        samples = arr[i:i+100] - dim_means
        sum_of_squares = (samples**2).sum(0)
        total_variance += (sum_of_squares/N)
    
    dim_stds = total_variance**0.5
    
    out_features = []
    start_idx = 0
    for l in lengths:
        out_features.append(
            arr[start_idx:start_idx+l] / dim_stds
        )
        start_idx += l
    
    return out_features


def add_certainties_to_data_dict(data_dict, alignment_paths):
    certainty_dict, seen_utterances = get_certainties_from_multiple_ctms(alignment_paths)
    new_data_dict = {k: [] for k in data_dict.keys()}
    new_data_dict['certainties'] = []
    for i, utt_id in enumerate(data_dict['utterance_segment_ids']):
        if utt_id in seen_utterances:
            for k in data_dict.keys():
                new_data_dict[k].append(data_dict[k][i])
            new_data_dict['certainties'].append(certainty_dict[utt_id])
    return new_data_dict


def data_dict_length_split(data_dict, max_len):
    # data_dict['mfcc'] = data_range_normalisation(data_dict['mfcc'])
    print('WARNING: data_dict_length_split no longer normalises data, only for splitting sequences')
    new_data_dict = {k: [] for k in data_dict.keys()}
    for j in range(len(data_dict['mfcc'])):
        feature_chunks = list(chunks(data_dict['mfcc'][j], max_len))
        new_data_dict['mfcc'].extend(feature_chunks)
        for k in set(data_dict.keys()) - {'mfcc'}:
            new_data_dict[k].extend(
                [data_dict[k][j] for _ in range(len(feature_chunks))]
            )
    return new_data_dict


def include_utterance(utt_id, n_frames, fi, oob):
    if utt_id in fi and utt_id not in oob and n_frames <= 400 and n_frames >= 50:
        return True
    else:
        return False


def features_from_time_bounds(_features, _start_time, dur, frames_per_second):
    num_prev_frames = frames_per_second * _start_time
    num_word_frames = frames_per_second * dur
    return _features[int(num_prev_frames) : int(num_prev_frames + num_word_frames)]


def filter_alignment_df(alignment_df, g, utt2dur, filter_func=include_utterance):

    finder = alignment_df[0].tolist()
    out_of_bounds_utt_ids = []

    for row in tqdm(alignment_df.iloc, disable=True):
        utt_id, _, start_time, word_duration, word, conf_score = row
        end_time = start_time + word_duration
        if end_time > utt2dur[utt_id]:
            out_of_bounds_utt_ids.append(utt_id)

    utt_names = [h[0] for h in g]
    features = [h[1] for h in g]
    num_frames = [len(f) for f in features]
    utt_durs = [float(utt2dur[u]) for u in utt_names]

    final_utt_names, final_features, final_num_frames, final_utt_durs = [], [], [], []

    for j, ut_id in tqdm(enumerate(utt_names), disable=True):
        if filter_func(ut_id, num_frames[j], finder, out_of_bounds_utt_ids):
            final_utt_names.append(utt_names[j])
            final_features.append(features[j])
            final_num_frames.append(num_frames[j])
            final_utt_durs.append(utt_durs[j])

    return final_utt_names, final_features, final_num_frames, final_utt_durs


def generate_data_features(
    final_utt_names, final_features, final_num_frames, final_utt_durs, alignment_df
):

    feature_lookup = {uid: final_features[j] for j, uid in enumerate(final_utt_names)}
    frames_per_second = np.mean(np.array(final_num_frames) / np.array(final_utt_durs))

    utt_ids, word_texts, word_mfcc = [], [], []

    for row in tqdm(alignment_df.iloc, disable=True):
        utt_id, _, start_time, word_duration, word, conf_score = row
        if utt_id not in final_utt_names:
            continue
        utt_ids.append(utt_id)
        word_texts.append(word)
        word_mfcc.append(
            features_from_time_bounds(
                feature_lookup[utt_id],
                float(start_time),
                float(word_duration),
                frames_per_second,
            )
        )

    return utt_ids, word_texts, word_mfcc


def generate_word_indices(utt_ids):
    word_indices = [0]
    m = 1
    for j, uid in enumerate(utt_ids[1:], 1):
        if uid == utt_ids[j - 1]:
            m += 1
        else:
            m = 0
        word_indices.append(m)
    return word_indices


def generate_data_dict_words(features_paths, alignment_path, utt2dur_path):
    with open(utt2dur_path, "r") as f:
        lines = f.read().split("\n")[:-1]
        utt2dur = {l.split()[0]: float(l.split()[1]) for l in lines}
    align_df = pd.read_csv(alignment_path, sep=" ", header=None)[range(6)]
    if isinstance(features_paths, str):
        gg = list(load_ark(features_paths))
    elif isinstance(features_paths, list):
        gg = []
        for fp in features_paths:
            gg.extend(list(load_ark(fp)))

    (
        final_utt_names,
        final_features,
        final_num_frames,
        final_utt_durs,
    ) = filter_alignment_df(align_df, gg, utt2dur)
    utt_ids, word_texts, word_mfcc = generate_data_features(
        final_utt_names, final_features, final_num_frames, final_utt_durs, align_df
    )
    word_indices = generate_word_indices(utt_ids)

    data_dict = {
        "utterance_segment_ids": utt_ids,
        "words": word_texts,
        "mfcc": word_mfcc,
        "word_index": word_indices,
    }

    return data_dict


def generate_data_dict_utt(features_paths, text_path=None):
    if isinstance(features_paths, str):
        gg = list(load_ark(features_paths))
    elif isinstance(features_paths, list):
        gg = []
        for fp in features_paths:
            gg.extend(list(load_ark(fp)))

    utt_ids = [h[0] for h in gg]
    utt_mfcc = [h[1] for h in gg]

    if text_path:
        with open(text_path, "r") as f:
            lines = f.read().split("\n")[-1]
        id_to_text = {l.split()[0]: " ".join(l.split()[1:]) for l in lines}
        texts = [id_to_text.get(utt_id) for utt_id in utt_ids]
    else:
        texts = [None for _ in utt_ids]

    return {"utterance_segment_ids": utt_ids, "text": texts, "mfcc": utt_mfcc}


def split_data_dict(data_dict, labelled_utt_list_path, unlabelled_utt_list_path):

    with open(labelled_utt_list_path, 'r') as f:
        labelled_utt_ids = set(f.read().split('\n')[:-1])

    with open(unlabelled_utt_list_path, 'r') as f:
        unlabelled_utt_ids = set(f.read().split('\n')[:-1])

    assert labelled_utt_ids.intersection(unlabelled_utt_ids) == set(), \
        'Labelled and unlabelled utt ids intersect!'

    labelled_data_dict = {k: [] for k in data_dict.keys()}
    unlabelled_data_dict = {k: [] for k in data_dict.keys()}
    
    seen_labelled_utt_ids, seen_unlabelled_utt_ids = set(), set()

    for i, utt_segment_id in enumerate(data_dict['utterance_segment_ids']):
        
        if utt_segment_id in labelled_utt_ids:
            seen_labelled_utt_ids.add(utt_segment_id)
            for k in data_dict.keys():
                labelled_data_dict[k].append(data_dict[k][i])

        if utt_segment_id in unlabelled_utt_ids:
            seen_unlabelled_utt_ids.add(utt_segment_id)
            for k in data_dict.keys():
                unlabelled_data_dict[k].append(data_dict[k][i])

    assert seen_labelled_utt_ids == labelled_utt_ids, 'Not all labelled utt ids seen in selected data'
    assert seen_unlabelled_utt_ids == unlabelled_utt_ids, 'Not all unlabelled utt ids seen in selected data'

    return labelled_data_dict, unlabelled_data_dict


def coll_fn_words(instances):
    audio_words = [torch.tensor(ins[0]) for ins in instances]
    text_words = [ins[1] for ins in instances]
    utt_ids = [ins[2] for ins in instances]
    word_indices = [ins[3] for ins in instances]
    packed_audio = nn.utils.rnn.pack_sequence(audio_words, enforce_sorted=False)
    padded, lengths = nn.utils.rnn.pad_packed_sequence(packed_audio, batch_first=True)
    return {
        "padded_features": padded,
        "lengths": lengths,
        "utterance_segment_ids": utt_ids,
        "word_indices": word_indices,
        "text_words": text_words,
    }


def coll_fn_utt(instances):
    audio = [torch.tensor(ins[0]) for ins in instances]
    texts = [ins[1] for ins in instances]
    utt_ids = [ins[2] for ins in instances]
    packed_audio = nn.utils.rnn.pack_sequence(audio, enforce_sorted=False)
    padded, lengths = nn.utils.rnn.pad_packed_sequence(packed_audio, batch_first=True)
    return {
        "padded_features": padded,
        "lengths": lengths,
        "utterance_segment_ids": utt_ids,
        "texts": texts,
    }


def coll_fn_utt_with_max_length(max_length):
    def ret_func(instances):
        audio = [torch.tensor(ins[0]) for ins in instances]
        texts = [ins[1] for ins in instances]
        utt_ids = [ins[2] for ins in instances]
        packed_audio = nn.utils.rnn.pack_sequence(audio, enforce_sorted=False)
        padded, lengths = nn.utils.rnn.pad_packed_sequence(packed_audio, batch_first=True, total_length=max_length)
        batch_size, *other_dims = padded.shape
        return {
            "padded_features": padded.reshape(len(instances), 1, *other_dims),
            "lengths": lengths,
            "utterance_segment_ids": utt_ids,
            "texts": texts,
        }
    return ret_func


def coll_fn_utt_with_targets(instances):
    audio = [torch.tensor(ins[0]) for ins in instances]
    texts = [ins[1] for ins in instances]
    utt_ids = [ins[2] for ins in instances]
    targets = torch.tensor([ins[3] for ins in instances])
    packed_audio = nn.utils.rnn.pack_sequence(audio, enforce_sorted=False)
    padded, lengths = nn.utils.rnn.pad_packed_sequence(packed_audio, batch_first=True)
    return {
        "padded_features": padded,
        "lengths": lengths,
        "utterance_segment_ids": utt_ids,
        "texts": texts,
        "targets": targets
    }


def generate_coll_fn_simclr_utt(transformation_distribution):
    def coll_fn_simclr_utt(instances):
        audio = [transformation_distribution(torch.tensor(ins[0])).T for ins in instances]
        audio += [transformation_distribution(torch.tensor(ins[0])).T for ins in instances]
        texts = [ins[1] for ins in instances] * 2
        utt_ids = [ins[2] for ins in instances] * 2
        packed_audio = nn.utils.rnn.pack_sequence(audio, enforce_sorted=False)
        padded, lengths = nn.utils.rnn.pad_packed_sequence(packed_audio, batch_first=True)
        return {
            "padded_features": padded,
            "lengths": lengths,
            "utterance_segment_ids": utt_ids,
            "texts": texts,
        }
    return coll_fn_simclr_utt


def get_certainties_from_ctm(ctm_path):
    with open(ctm_path, 'r') as f:
        lines = f.read().split('\n')[:-1]
    certainties_dict = {}
    durations_dict = {}
    for line in lines:
        uid, _, _, dur, _, certainty = line.split()
        certainties_dict[uid] = certainties_dict.get(uid, []) + [float(certainty)]
        durations_dict[uid] = durations_dict.get(uid, []) + [float(dur)]
    target_dict = {}
    for uttid, durations in durations_dict.items():
        target_dict[uttid] = (np.array(durations)@np.array(certainties_dict [uttid]))/sum(durations)
    return target_dict


def get_certainties_from_multiple_ctms(alignment_paths):
    all_included_utts = set()
    certainty_dict = {}
    for alignment_path in alignment_paths:
        new_certainty_dict = get_certainties_from_ctm(alignment_path)
        utts_in_file = set(new_certainty_dict.keys())
        assert utts_in_file.intersection(all_included_utts) == set()
        all_included_utts = all_included_utts.union(utts_in_file)
        certainty_dict.update(new_certainty_dict)
    return certainty_dict, all_included_utts