import json
import os
from numpy import pi, unique
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
from torch.distributions.categorical import Categorical

from .data_utils import *

TQDM_MODE = True


class Index:
    def __init__(self, dataset):
        self.dataset = dataset
        self.number_partially_labelled_instances = 0
        self.labelled_idx = None
        self.unlabelled_idx = None
        self.temp_labelled_idx = None

    def label_instance(self, i):
        raise NotImplementedError

    def label_window(self, window):
        raise NotImplementedError

    def temporarily_label_window(self, window):
        raise NotImplementedError

    def new_window_unlabelled(self, new_window):
        raise NotImplementedError

    def is_partially_labelled(self, i):
        return NotImplementedError

    def is_partially_temporarily_labelled(self, i):
        return NotImplementedError

    def has_any_labels(self, i):
        return bool(self.is_partially_labelled(i)) or bool(
            self.is_partially_temporarily_labelled(i)
        )

    def is_labelled(self, i):
        NotImplementedError

    def is_partially_unlabelled(self, i):
        NotImplementedError

    def get_number_partially_labelled_instances(self):
        return self.number_partially_labelled_instances

    def __getitem__(self, item):

        if isinstance(item, int):
            idx = [item]
        elif isinstance(item, slice):
            idx = list(range(*item.indices(len(self.labelled_idx))))
        elif isinstance(item, list):
            idx = item
        else:
            raise TypeError(f"Cannot index SentenceIndex with type {type(item)}")

        return {
            i: {
                "labelled_idx": self.labelled_idx[i],
                "unlabelled_idx": self.unlabelled_idx[i],
                "temp_labelled_idx": self.temp_labelled_idx[i],
            }
            for i in idx
        }


class DimensionlessIndex(Index):
    def __init__(self, dataset):
        super(DimensionlessIndex, self).__init__(dataset)
        self.labelled_idx = {j: False for j in range(len(dataset.data))}
        self.unlabelled_idx = {j: True for j, d in enumerate(dataset.data)}
        self.temp_labelled_idx = {j: False for j in range(len(dataset.data))}\

    def is_partially_labelled(self, i):
        return bool(self.labelled_idx[i])

    def is_partially_temporarily_labelled(self, i):
        return bool(self.temp_labelled_idx[i])

    def is_labelled(self, i):
        return not bool(self.unlabelled_idx[i])

    def is_partially_unlabelled(self, i):
        return bool(self.unlabelled_idx[i])

    def label_instance(self, i):
        self.number_partially_labelled_instances += 1
        self.labelled_idx[i] = True
        self.unlabelled_idx[i] = False

    def label_window(self, window):
        if not isinstance(window, DimensionlessAnnotationUnit):
            raise TypeError("DimensionlessIndex requires DimensionlessAnnotationUnit")
        self.label_instance(window.i)

    def temporarily_label_window(self, window):
        self.unlabelled_idx[window.i] = False
        self.temp_labelled_idx[window.i] = True

    def new_window_unlabelled(self, new_window):
        return not self.labelled_idx[new_window.i]

    def save(self, save_path):
        with open(os.path.join(save_path, "agent_index.pk"), "w") as f:
            json.dump(
                {
                    "labelled_idx": self.labelled_idx,
                    "unlabelled_idx": self.unlabelled_idx,
                    "temporarily_labelled_idx": self.temp_labelled_idx,
                },
                f,
            )


class SentenceIndex(Index):
    def __init__(self, dataset):
        super(SentenceIndex, self).__init__(dataset)
        self.labelled_idx = {j: set() for j in range(len(dataset))}
        self.unlabelled_idx = {
            j: set(range(len(d))) for j, d in enumerate(dataset)
        }
        self.temp_labelled_idx = {j: set() for j in range(len(dataset))}

    def label_instance(self, i):
        if not self.is_partially_labelled(i):
            self.number_partially_labelled_instances += 1
        self.labelled_idx[i] = set(range(len(self.dataset[i])))
        self.unlabelled_idx[i] = set()

    def label_window(self, window):
        if not self.labelled_idx[window.i] and window.size > 0:
            self.number_partially_labelled_instances += 1
        self.labelled_idx[window.i].update(range(*window.bounds))
        self.unlabelled_idx[window.i] -= window.get_index_set()
        self.temp_labelled_idx[window.i] -= window.get_index_set()

    def temporarily_label_window(self, window):
        self.unlabelled_idx[window.i] -= window.get_index_set()
        self.temp_labelled_idx[window.i].update(window.get_index_set())

    def is_partially_labelled(self, i):
        return bool(len(self.labelled_idx[i]))

    def is_partially_temporarily_labelled(self, i):
        return bool(len(self.temp_labelled_idx[i]))

    def is_labelled(self, i):
        return not bool(len(self.unlabelled_idx[i]))

    def is_partially_unlabelled(self, i):
        return bool(len(self.unlabelled_idx[i]))

    def new_window_unlabelled(self, new_window):
        if new_window.get_index_set().intersection(self.labelled_idx[new_window.i]):
            return False
        else:
            return True

    def save(self, save_path):
        with open(os.path.join(save_path, "agent_index.pk"), "w") as f:
            json.dump(
                {
                    "labelled_idx": {k: list(v) for k, v in self.labelled_idx.items()},
                    "unlabelled_idx": {
                        k: list(v) for k, v in self.unlabelled_idx.items()
                    },
                    "temporarily_labelled_idx": {
                        k: list(v) for k, v in self.temp_labelled_idx.items()
                    },
                },
                f,
            )


class AnnotationUnit:
    def __init__(self, data_index, bounds, score):
        self.i = data_index
        self.bounds = bounds
        self.score = score
        self.slice = None

    def savable(self):
        return [self.i, self.bounds, self.score]

    def get_index_set(self):
        raise NotImplementedError

    def get_cost(self, dataset):
        self.cost = dataset.get_cost_by_window(self)


class SentenceSubsequenceAnnotationUnit(AnnotationUnit):
    def __init__(self, data_index, bounds, score):
        super(SentenceSubsequenceAnnotationUnit, self).__init__(data_index, bounds, score)
        self.size = bounds[1] - bounds[0]
        self.slice = slice(*bounds)

    def savable(self):
        return [int(self.i), list(map(int, self.bounds)), float(self.score)]

    def get_index_set(self):
        return set(range(*self.bounds))


class DimensionlessAnnotationUnit(AnnotationUnit):
    def __init__(self, data_index, bounds, score):
        super(DimensionlessAnnotationUnit, self).__init__(data_index, bounds, score)
        self.size = 1
        self.slice = ...

    def savable(self):
        return [int(self.i), None, float(self.score)]

    def get_index_set(self):
        return {0}
