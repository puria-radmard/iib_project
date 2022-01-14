import torch
from torch.nn import functional as F
from active_learning.acquisition.daf_metrics import *
from active_learning.acquisition.batch_acquisition import GraphCutWeightedCoreSetBatchAcquisition

metric_functions = {
    'image_reconstruction': AverageOfReconstructionLossesMetric,
    'l1': ReconstructionLossOfAverageMetric,
    'l2': AverageOfReconstructionLossesMetric,
    'l3': ReconstructionDisagreementMetric,
    'classification': LabelledRankMetric,
    'classification_with_graph_cut': GraphCutWeightedCoreSetBatchAcquisition
}