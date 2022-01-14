import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from functools import lru_cache
import json
from tqdm import tqdm
# from sklearn.cluster import KMeans
from tqdm.std import TqdmDefaultWriteLock
from ..annotation_classes import DimensionlessAnnotationUnit

TQDM_MODE = True

device = (
    torch.device('cuda') if torch.cuda.is_available() 
    else torch.device('cpu')
)


class AcquisitionAggregation:
    def __init__(self, functions, dataset):
        self.functions = functions
        self.dataset = dataset

    def aggregation_step(self):
        # This is for learning aggregations, e.g. LSA bandit set up
        raise NotImplementedError

    def acquisition_aggregation(self, scores):
        raise NotImplementedError

    def score(self, batch_indices):
        scores = []
        for function in self.functions:
            scores.append(function.score(i))
        return self.acquisition_aggregation(scores).reshape(-1)

    def step(self):
        self.aggregation_step()
        for function in self.functions:
            if isinstance(function, UnitwiseAcquisition):
                pass
            elif isinstance(function, DataAwareAcquisition):
                function.step()
            else:
                raise NotImplementedError(f"{type(function)} not a function type")
        pass


class SimpleAggregation(AcquisitionAggregation):
    def __init__(self, functions, dataset, weighting):
        self.weighting = F.normalize(weighting)
        super(SimpleAggregation, self).__init__(functions, dataset)

    def acquisition_aggregation(self, scores):
        return self.weighting @ scores


class LearningAggregation(AcquisitionAggregation):
    def __init__(self, functions, dataset):
        super(LearningAggregation, self).__init__(functions, dataset)
        raise NotImplementedError

    def acquisition_aggregation(self, scores):
        raise NotImplementedError

    def aggregation_step(self):
        raise NotImplementedError


class DataAwareAcquisition:
    def __init__(self, dataset):
        self.dataset = dataset

    def score(self, batch_indices):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


class UnitwiseAcquisition:
    def __init__(self, dataset):
        self.dataset = dataset

    def score(self, batch_indices):
        raise NotImplementedError


class BatchAcquisition:
    """No scores are used for this class of acquisition functions -"""

    def __init__(self, dataset):
        self.dataset = dataset

    def select_next_subset(self, candidate_windows, total_cost):
        raise NotImplementedError
        

class ModelBasedAcquision:
    """Provided with a model that is trained at each step"""
    
    def __init__(self, dataset, model, input_attr):
        self.model = model
        self.dataset = dataset
        self.input_attr = input_attr

    def score(self, batch_indices):
        input = self.dataset.__getitem__(self.input_attr)[batch_indices]
        return self.model(input)
