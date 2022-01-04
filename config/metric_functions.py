import torch
from torch.nn import functional as F
from active_learning.acquisition.daf_metrics import *

metric_functions = {
    'image_reconstruction': AverageOfReconstructionLossesMetric,
    'l1': ReconstructionLossOfAverageMetric,
    'l2': AverageOfReconstructionLossesMetric,
    'l3': ReconstructionDisagreementMetric,
    'classification': LabelledRankMetric
}