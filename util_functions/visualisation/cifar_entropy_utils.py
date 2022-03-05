import torch, os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100

cifar100_embeddings_path = '/home/alta/BLTSpeaking/exp-pr450/data/byol_embeddings_100.pkl'
cifar100_embeddings = torch.load(cifar100_embeddings_path)

cifar100_dataset = CIFAR100('data', train=True)

def get_labels_and_embeddings_by_indices(indices, embeddings=cifar100_embeddings, dataset=cifar100_dataset):
    
    # Index embeddings from tensor
    embs = embeddings[indices]

    # Access labels from dataset
    all_labels = dataset.targets

    # Index labels from dataset using indices
    labels = [all_labels[i] for i in indices]

    return labels, embs


def indices_from_path(path):
    with open(path, 'r') as f:
        lines = f.read()
    str_indices = lines.split('\n')[:-1]
    return list(map(int, str_indices))


def labels_and_embeddings_by_paths(paths, embeddings=cifar100_embeddings, dataset=cifar100_dataset):

    # We generalise with a list of paths, but if only one is provided then make it friendly
    if isinstance(paths, str):
        paths = [paths]

    # Initialise labels and embeds list
    all_labels = []
    all_embeds = []

    # Iterate over paths and collect both
    for path in paths:
        indices = indices_from_path(path)
        labels, embs = get_labels_and_embeddings_by_indices(indices, embeddings, dataset)

        # Labels is a list so update 'online'
        all_labels.extend(labels)
        all_embeds.append(embs)

    # Join all embeds for output
    all_embeds = torch.cat(all_embeds)

    return all_labels, all_embeds


def get_labelled_set_path(round_history_num, round_num):
    # Get just the one label path
    basemost_path = "/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_labelled_classification_recalibration"
    labelled_set_template = os.path.join(basemost_path, "round_history-{}/labelled_set_{}.txt")
    return labelled_set_template.format(round_history_num, round_num)


def get_cumulative_labelled_set_paths(round_history_num, round_num):
    # Get all label paths up to and including this one
    f = lambda i: get_labelled_set_path(round_history_num, i)
    return list(map(f, range(1, round_num + 1)))


def get_entropy(label_list):
    label_ids, label_counts = np.unique(label_list, return_counts = True)
    label_props = label_counts/label_counts.sum()
    entropy = - label_props.T @ np.log(label_props)
    return entropy