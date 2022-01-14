import numpy as np
from kaldiio import load_ark
import torch
from torch import nn
from sklearn.model_selection import train_test_split

__all__ = [
    "chunks",
    "coll_fn_utt_with_channel_insersion",
    "data_dict_length_split",
    "add_certainties_to_data_dict",
    "generate_data_dict_utt",
    "split_data_dict_by_labelled",
    "coll_fn_utt",
    'train_test_split_data_dict',
    'combine_data_dicts'
]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def add_certainties_to_data_dict(data_dict, alignment_paths):
    certainty_dict, seen_utterances = get_certainties_from_multiple_ctms(
        alignment_paths
    )
    new_data_dict = {k: [] for k in data_dict.keys()}
    new_data_dict["certainties"] = []
    for i, utt_id in enumerate(data_dict["utterance_segment_ids"]):
        if utt_id in seen_utterances:
            for k in data_dict.keys():
                new_data_dict[k].append(data_dict[k][i])
            new_data_dict["certainties"].append(certainty_dict[utt_id])
    return new_data_dict


def data_dict_length_split(data_dict, max_len):
    # data_dict['mfcc'] = data_range_normalisation(data_dict['mfcc'])
    print(
        "WARNING: data_dict_length_split no longer normalises data, only for splitting sequences"
    )
    new_data_dict = {k: [] for k in data_dict.keys()}
    for j in range(len(data_dict["mfcc"])):
        feature_chunks = list(chunks(data_dict["mfcc"][j], max_len))
        new_data_dict["mfcc"].extend(feature_chunks)
        for k in set(data_dict.keys()) - {"mfcc"}:
            new_data_dict[k].extend(
                [data_dict[k][j] for _ in range(len(feature_chunks))]
            )
    return new_data_dict


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


def split_data_dict_by_labelled(data_dict, labelled_utt_list_path, unlabelled_utt_list_path):

    if labelled_utt_list_path is not None:
        with open(labelled_utt_list_path, "r") as f:
            labelled_utt_ids = set(f.read().split("\n")[:-1])
    else:
        labelled_utt_ids = set()

    with open(unlabelled_utt_list_path, "r") as f:
        unlabelled_utt_ids = set(f.read().split("\n")[:-1])

    assert (
        labelled_utt_ids.intersection(unlabelled_utt_ids) == set()
    ), "Labelled and unlabelled utt ids intersect!"

    labelled_data_dict = {k: [] for k in data_dict.keys()}
    unlabelled_data_dict = {k: [] for k in data_dict.keys()}

    seen_labelled_utt_ids, seen_unlabelled_utt_ids = set(), set()

    for i, utt_segment_id in enumerate(data_dict["utterance_segment_ids"]):

        if utt_segment_id in labelled_utt_ids:
            seen_labelled_utt_ids.add(utt_segment_id)
            for k in data_dict.keys():
                labelled_data_dict[k].append(data_dict[k][i])

        if utt_segment_id in unlabelled_utt_ids:
            seen_unlabelled_utt_ids.add(utt_segment_id)
            for k in data_dict.keys():
                unlabelled_data_dict[k].append(data_dict[k][i])

    assert (seen_labelled_utt_ids <= labelled_utt_ids)
    print(len(labelled_utt_ids) - len(seen_labelled_utt_ids), "labelled utt ids not seen in selected data")
    assert (seen_unlabelled_utt_ids <= unlabelled_utt_ids)
    print(len(unlabelled_utt_ids) - len(seen_unlabelled_utt_ids), "unlabelled utt ids not seen in selected data")

    return labelled_data_dict, unlabelled_data_dict


def coll_fn_utt(instances):
    res = {}
    for k in instances[0].keys():
        inses = [ins[k] for ins in instances]
        try: res[k] = torch.tensor(inses)
        except: res[k] = inses
    audio = [torch.tensor(ins["audio"]) for ins in instances]
    packed_audio = nn.utils.rnn.pack_sequence(audio, enforce_sorted=False)
    padded, lengths = nn.utils.rnn.pad_packed_sequence(packed_audio, batch_first=True)
    res["padded_features"] = padded
    return res


def coll_fn_utt_with_channel_insersion(instances):
    res = coll_fn_utt(instances)
    res["padded_features"] = res["padded_features"][:,None]
    return res


def get_certainties_from_ctm(ctm_path):
    with open(ctm_path, "r") as f:
        lines = f.read().split("\n")[:-1]
    certainties_dict = {}
    durations_dict = {}
    for line in lines:
        uid, _, _, dur, _, certainty = line.split()
        certainties_dict[uid] = certainties_dict.get(uid, []) + [float(certainty)]
        durations_dict[uid] = durations_dict.get(uid, []) + [float(dur)]
    target_dict = {}
    for uttid, durations in durations_dict.items():
        target_dict[uttid] = (
            np.array(durations) @ np.array(certainties_dict[uttid])
        ) / sum(durations)
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


def train_test_split_data_dict(data_dict, test_prop):
    keys = list(data_dict.keys())
    split_values = train_test_split(*[data_dict[k] for k in keys], test_size=test_prop)
    new_values = list(zip(*(iter(split_values), ) * 2))
    train_dict = {k: new_values[i][0] for i, k in enumerate(keys)}
    test_dict = {k: new_values[i][1] for i, k in enumerate(keys)}
    return train_dict, test_dict


def combine_data_dicts(*dicts):
    assert all(d.keys() == dicts[0].keys() for d in dicts)
    out_dict = {k: [] for k in dicts[0].keys()}
    for d in dicts:
        [out_dict[k].extend(d[k]) for k in d.keys()]
    return out_dict


