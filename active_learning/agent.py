import logging
import random
from random import sample
from typing import List, Dict, NoReturn
import os, sys
import numpy as np
import pandas as pd
import json
from torch.utils import data
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler
from tqdm import tqdm

from .util_classes import RenovationError
from .data_utils import command_line

TQDM_MODE = True


class AgentBase:
    def __init__(
            self, train_set, batch_size, selector_class, model, device, budget
    ):
        self.batch_size = batch_size
        self.train_set = train_set
        self.selector = selector_class
        self.selector.assign_agent(self)
        self.model = model
        self.device = device

        self.budget = budget
        self.initial_budget = self.budget

        self.unlabelled_set = None
        self.labelled_set = None
        self.num = 0
        self.round_all_word_scores = {}

    def init(self, init_cost_prop, seed=42):
        print("starting random init")
        self.random_init(init_cost_prop, seed)
        self.update_datasets()
        print("finished random init")

    def step(self, update_dataset=True):
        print('step')
        if update_dataset:
            self.update_dataset_attributes()
        self.update_index()
        self.update_datasets()

    def budget_spent(self):
        return self.initial_budget - self.budget

    def num_instances(self):
        return sum([len(l) for l in self.labelled_set])

    def save(self, save_path):
        self.train_set.index.save(save_path)
        self.selector.save(save_path)
        with open(os.path.join(save_path, "all_word_scores_no_nan.json"), "w") as f:
            json.dump(self.round_all_word_scores, f)
        raise RenovationError('This is original project specific')

    def random_init(self, init_cost_prop, seed):
        """
        Randomly initialise self.labelled_idx dictionary
        """

        # Randomly order the datapoints in the dataset using a seed
        init_sampler = random.Random(seed)
        randomly_ordered_indices = sorted(range(len(self.train_set)), key=lambda k: init_sampler.random())

        # Use dataset property:
        init_cost = self.train_set.total_cost * init_cost_prop

        # Go through random order, stop when cost hit
        budget_spent = 0
        for i in randomly_ordered_indices:
            self.train_set.index.label_instance(i)
            budget_spent += self.train_set.get_cost_by_index(i)
            if budget_spent > init_cost:
                break

        # 'Global' cost budget variable
        self.budget -= budget_spent

        print(f"total dataset cost: {self.train_set.total_cost}")
        print(f"initialised with {budget_spent} cost  |   remaining cost budget: {self.budget}")

    def __iter__(self):
        return self

    def __next__(self):
        num = self.num
        self.num += 1
        if num < 0:
            raise StopIteration
        if num > 0:
            self.step()
        if self.budget <= 0:
            self.num = -1
        return self.budget

    def update_datasets(self):
        unlabelled_instances = set()
        labelled_instances = set()

        print("update datasets")
        for i in tqdm(range(len(self.train_set)), disable=not TQDM_MODE):
            if self.train_set.index.is_partially_unlabelled(i):
                unlabelled_instances.add(i)
            if self.train_set.index.has_any_labels(i):
                labelled_instances.add(i)

        self.unlabelled_set = list(
            BatchSampler(
                SubsetRandomSampler(list(unlabelled_instances)),
                self.batch_size,
                drop_last=False,
            )
        )

        self.labelled_set = list(
            BatchSampler(
                SubsetRandomSampler(list(labelled_instances)),
                self.batch_size,
                drop_last=False,
            )
        )

    def update_dataset_attributes(self):
        """
        Score unlabelled instances in terms of their suitability to be labelled next.
        Add the highest scoring instance indices in the dataset to self.labelled_idx
        """

        if self.budget <= 0:
            logging.warning("no more budget left!")

        with torch.no_grad():
            # print('get sentence scores')
            for batch_indices in tqdm(
                self.unlabelled_set + self.labelled_set, disable=not TQDM_MODE
            ):
                instances, _, lengths, _ = self.train_set.get_batch(batch_indices, labels_important=False)
                try:
                    model_attrs = self.model(instances.to(self.device))
                except:
                    model_attrs = self.model(instances)
                model_attrs = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in model_attrs.items()}
                self.train_set.update_attributes(batch_indices, model_attrs, lengths)

    def update_index(self):

        all_windows = []
        for labelled_batch_indices in tqdm(self.labelled_set, disable=not TQDM_MODE):
            windows = self.selector.window_generation(labelled_batch_indices, self.train_set)
            all_windows.extend(windows)
        for unlabelled_batch_indices in tqdm(self.unlabelled_set, disable=not TQDM_MODE):
            windows = self.selector.window_generation(unlabelled_batch_indices, self.train_set)
            all_windows.extend(windows)

        all_windows.sort(key=lambda e: e.score, reverse=True)
        best_windows, budget_spent = self.selector.select_best(all_windows)
        self.budget -= budget_spent
        if self.budget < 0:
            logging.warning("no more budget left!")

        total_cost = 0
        for window in best_windows:
            total_cost += window.cost
            self.train_set.index.label_window(window)

        # No more windows of this size left
        if total_cost < self.selector.round_cost:
            self.selector.reduce_window_size()


class ActiveLearningAgent(AgentBase):
    def __init__(self, train_set, batch_size, selector_class, model, device, budget):
        super(ActiveLearningAgent, self).__init__(train_set, batch_size, selector_class, model, device, budget)
        # ADD AN EXCEPTION FOR THE WRONG TYPE OF SELECTOR HERE


class SubsetSelectionAgent(AgentBase):
    def __init__(self, train_set, batch_size, selector_class, model, device, budget):
        super(SubsetSelectionAgent, self).__init__(train_set, batch_size, selector_class, model, device, budget)


class KaldiAgent(AgentBase):
    def __init__(self, train_set, batch_size, selector_class, model, device, namer, budget, suffix):
        super(KaldiAgent, self).__init__(train_set, batch_size, selector_class, model, device, budget)
        command_line('bash path.sh')
        self.namer = namer
        self.suffix = suffix

    def init(self, init_parts_path):
        print(f'Initialising agent from {init_parts_path}')
        with open(init_parts_path, "r") as f:
            lines = f.read()[:-1].split('\n')
            lines = [l.rstrip() for l in lines]
        previously_selected_indices = []

        # TEMPORARY FIX - THIS CAN BE MADE QUICKER UNDER A PERMANENT BY PARTS ASSUMPTION
        utt_ids = {}
        [utt_ids.update({u: (j, k) for k, u in enumerate(self.train_set.utt_ids[j])}) for j in range(len(self.train_set))]
        utt_idxs = []
        for line in lines:
            idxs = utt_ids.get(line)
            if idxs:
                a, b = idxs
                previously_selected_indices.append(a)
                utt_idxs.append(b)
        budget_previously_spent = 0
        for i, j in enumerate(previously_selected_indices):
            self.train_set.index.label_instance(j)
            # Fix this when things are resolved
            budget_previously_spent += self.train_set.cost[j][utt_idxs[i]]

        # self.budget -= budget_previously_spent
        print(f'Finished initialising agent with {budget_previously_spent} data, NOT included in {self.budget} budget')

        self.update_datasets(write_to_file = False)

    def step(self, update_dataset=True):
        self.namer.update_paths(self.suffix)
        return super().step(update_dataset=update_dataset)

    def random_init(self, num_instances, seed):
        raise NotImplementedError("Random init not implemented for KaldiAgent, init requires a utterance list file")

    def update_datasets(self, write_to_file = True):

        unlabelled_instances = set()
        labelled_instances = set()

        print("update datasets")
        for i in tqdm(range(len(self.train_set)), disable=not TQDM_MODE):
            if self.train_set.index.is_partially_unlabelled(i):
                unlabelled_instances.add(i)
            if self.train_set.index.has_any_labels(i):
                labelled_instances.add(i)

        self.unlabelled_set = list(
            BatchSampler(
                SubsetRandomSampler(list(unlabelled_instances)),
                self.batch_size,
                drop_last=False,
            )
        )

        self.labelled_set = list(
            BatchSampler(
                SubsetRandomSampler(list(labelled_instances)),
                self.batch_size,
                drop_last=False,
            )
        )

        if write_to_file:

            with open(self.namer.current_paths['labelled_utts'], 'w') as f:
                # is sorting an issue? Kaldi might have a case for this
                # MAKE SURE THIS DOES AFFECT LABELLED_SET
                for group_index in labelled_instances:
                    group = self.train_set.utt_ids[group_index]
                    labelled_utt_ids = [g for i, g in enumerate(group) if i in self.train_set.index.labelled_idx[group_index]]
                    for utt_id in labelled_utt_ids:
                        f.write(utt_id)
                        f.write('\n')

            with open(self.namer.current_paths['unlabelled_utts'], 'w') as f:
                # is sorting an issue? Kaldi might have a case for this
                for group_index in unlabelled_instances:
                    group = self.train_set.utt_ids[group_index]
                    unlabelled_utt_ids = [g for i, g in enumerate(group) if i in self.train_set.index.unlabelled_idx[group_index]]
                    for utt_id in unlabelled_utt_ids:
                        f.write(utt_id)
                        f.write('\n')

            clustersize = 30

            for feature_name in self.namer.feature_names:
                break
                self.prepare_feat_subset(feature_name)
                combined_feature_name = self.combine_feat_subset(feature_name)
                self.compute_cmvn(combined_feature_name)
                self.split_data_dir(combined_feature_name, clustersize)
                self.split_data_dir(feature_name, clustersize)


class AutomaticKaldiNameIncrementer:

    def __init__(self, model_run_dir:str, data_run_dir:str, log_run_dir:str, base_model_paths:dict, base_data_paths:dict, base_log_paths:dict, constant_paths:dict, feature_names:list, make=True):
        self.model_run_dir, self.data_run_dir, self.log_run_dir = model_run_dir, data_run_dir, log_run_dir
        self.previous_paths = {}
        self.previous_paths.update({k: os.path.join(model_run_dir, v.split('/')[-1]) for k, v in base_model_paths.items()})
        self.previous_paths.update({k: os.path.join(data_run_dir, v.split('/')[-1]) for k, v in base_data_paths.items()})
        self.previous_paths.update({k: os.path.join(data_run_dir, v.split('/')[-1]) for k, v in base_log_paths.items()})

        if make:

            for dir in [data_run_dir, model_run_dir, log_run_dir]:
                try:
                    os.mkdir(dir)
                except FileExistsError:
                    non_symb = [f for f in os.listdir(dir) if not os.path.islink(os.path.join(dir, f))]
                    if len(non_symb) == 0:
                        print(f"{dir} exists but only has symbolic links, ignoring error")
                    else:
                        raise FileExistsError()

            for file_path in base_model_paths.values():
                sym_path = os.path.join(model_run_dir, file_path.split('/')[-1])
                try:
                    os.symlink(file_path, sym_path)
                except FileExistsError:
                    assert os.readlink(sym_path) == file_path

            for file_path in base_data_paths.values():
                sym_path = os.path.join(data_run_dir, file_path.split('/')[-1])
                try:
                    os.symlink(file_path, sym_path)
                except FileExistsError:
                    assert os.readlink(sym_path) == file_path

            for file_path in base_log_paths.values():
                sym_path = os.path.join(log_run_dir, file_path.split('/')[-1])
                try:
                    os.symlink(file_path, sym_path)
                except FileExistsError:
                    assert os.readlink(sym_path) == file_path

        self.current_paths = self.previous_paths.copy()
        self.constant_paths = constant_paths
        self.feature_names = feature_names

    def update_paths(self, suffix):
        self.previous_paths = self.current_paths
        self.current_paths = {k: v+suffix for k, v in self.previous_paths.items()}
