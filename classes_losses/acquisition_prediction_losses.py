import math
import sys

import numpy as np
from util_functions.jacobian import fit_beta, jacobian_transform
from torch import nn
import torch
from config import device
from torchsort import soft_rank


class SpearmanRankLoss(nn.Module):
    """
        Spearmans rank loss: https://github.com/teddykoker/torchsort
        
        This loss takes the Soft Rank (Blondel et al. 2020) of both the target and the input,
        meaning all comparison between the two ranks becomes differentiable.
        
        It then returns a negative dot product between the two, to be minimised.
        See 
    """

    def __init__(self, regularization="l2", regularization_strength=1.0):
        super(SpearmanRankLoss, self).__init__()
        self.regularization=regularization  # l2 or kl
        self.regularization_strength=regularization_strength  # epsilon in paper

    def soft_rank(self, x):
        return soft_rank(x, self.regularization, self.regularization_strength)

    @staticmethod
    def normalise_vector(x):
        x = x - x.mean()
        x = x / x.norm()
        return x

    def forward(self, input, target):
        
        # Reshape for interface
        target = target.reshape(1, -1)
        input = input.reshape(1, -1)

        # Take soft rank of both vectors
        input = self.soft_rank(input)
        target = self.soft_rank(target)

        # Normalise the two soft ranks
        input = self.normalise_vector(input)
        target = self.normalise_vector(target)

        # Return 
        return - (input * target).sum()


class LogNormalMSELoss(nn.MSELoss):

    def __init__(self, reduction = 'mean'):
        super(LogNormalMSELoss, self).__init__(reduction=reduction)

    def forward(self, input, target):
        log_target = torch.log(target)
        input = input.reshape(target.shape)
        return super(LogNormalMSELoss, self).forward(input, log_target)


class BCELossWithTransform(nn.BCELoss):

    def _transform_target(self, target):
        raise NotImplementedError

    def forward(self, input, target):
        # Transform the data, e.g. make it [0, 1] scale
        transformed_target = self._transform_target(target).to(target.dtype)
        # Reshape the input to make sure loss works
        input = input.reshape(transformed_target.shape)
        return super(BCELossWithTransform, self).forward(input, transformed_target)


class RescaleBCELoss(BCELossWithTransform):

    def __init__(self, log_C, reduction = 'mean'):
        self.max_val = log_C.float()
        super(RescaleBCELoss, self).__init__(reduction=reduction)
    
    def _transform_target(self, target):
        # Bring to [0, 1] scale
        transformed_target = target / self.max_val
        return transformed_target


class SqrtExponentialJacobianTransformedBCELoss(BCELossWithTransform):

    def __init__(self, train_targets, lr, beta_0, num_steps=500, base_transform = lambda x: x+1., reduction = 'mean'):

        # Infer the maximum likelihood beta
        data = base_transform(train_targets)
        self.beta, self.beta_history, self.llh_history, neg_C = fit_beta(data, beta_0, lr, num_steps)

        # Log this for later!
        print("Fitted beta:\t", self.beta)

        # Need to do the default one to targets for LC
        self.base_transform = base_transform

        super(SqrtExponentialJacobianTransformedBCELoss, self).__init__(reduction=reduction)

    def _transform_target(self, target):
        # Pass the data through the learned Jacobian
        targets = self.base_transform(target)
        transformed_target = jacobian_transform(targets, self.beta, 1)
        return transformed_target


class NonParametricJacobianTransformedBCELoss(BCELossWithTransform):
    
    def __init__(self, train_targets, num_bins=500, base_transform = lambda x: x+1., reduction = 'mean'):

        self.start_data = base_transform(train_targets).to('cpu')
        
        # Need to do the default one to targets for LC
        self.base_transform = base_transform

        # self.bins[i] now ENDS the (i-1)th bin. 
        # i.e. self.hist[j] contains bar bounded by (self.bins[j-1], self.bins[j])
        np_bins = np.linspace(0, 1, num_bins)
        np_counts = np.histogram(self.start_data.numpy(), np_bins, density = True)[0]
        self.bins, self.counts = torch.tensor(np_bins, dtype=self.start_data.dtype), torch.tensor(np_counts, dtype=self.start_data.dtype)
        self.hist = self._concat_element(self.counts)

        # Widths of each bin, should be the same!
        _deltas = torch.tensor(np.diff(np_bins), dtype=self.start_data.dtype)
        self.deltas = self._concat_element(_deltas, torch.tensor(math.inf)) # torch.inf

        # Get the CDF now instead of having to compute it
        # i.e. if x > self.bins[i] then it has full pars self.cdf[i] + interpolation
        prob_weights_with_nan = self.hist * self.deltas
        prob_weights_with_nan[torch.isnan(prob_weights_with_nan)] = 0.
        self.prob_weights = prob_weights_with_nan
        # self.prob_weights = torch.nan_to_num(self.hist * self.deltas)
        
        self.cdf = torch.tensor([self.prob_weights[:i].sum() for i in range(self.prob_weights.size(0) + 1)])

        super(NonParametricJacobianTransformedBCELoss, self).__init__(reduction=reduction)

    def remove_from_cdf(self, negated_targets):
        # If you want to rmove a portion of the dataset from the cdf, you can remove their effect 
        # on each of the object's attributes
        pass # TODO


    @staticmethod
    def _concat_element(tensor, element = 0.):
        return torch.cat([tensor, torch.tensor([element])])

    def _transform_target(self, target):

        target = self.base_transform(target).to('cpu')

        # Sum up all the bins before the target, then get an interpolation for this one
        # Which bins are these targets from?        
        bin_idxs = (self.bins.repeat(len(target), 1) <= target.reshape(-1, 1)).sum(-1) - 1

        # CDF up to bin boundary
        full_bins = self.cdf[bin_idxs]

        # Interpolate the last bin
        start_boundaries = self.bins[bin_idxs]
        _deltas = self.deltas[bin_idxs]
        fractions = (target - start_boundaries) / _deltas
        part_bins = self.prob_weights[bin_idxs] * fractions

        capital_f_targets = full_bins + part_bins

        return capital_f_targets.to(device)

        
class AdditionImportanceWeightedBCELoss(nn.BCELoss):

    def __init__(self, bce_loss_object, num_bins, fix_after):

        # Initialise parent class and the transformer, but make sure it does 
        # not aggregate loss, which we will weight per instane in batch
        super(AdditionImportanceWeightedBCELoss, self).__init__(reduction='none')
        self.loss = bce_loss_object

        # Set the boundaries of each bin that is weighted
        self.bin_boundaries = torch.linspace(0, 1, num_bins).to('cpu')

        # Number of instances we've seen in each bin
        self.bin_counts = torch.zeros_like(self.bin_boundaries)[1:].to('cpu')
        # Number of instances we've seen, i.e. batch_count * batch_size
        self.total_count = 0.

        ## i.e. online version of algorithm (float allows torch.inf too)
        if isinstance(fix_after, int) or isinstance(fix_after, float):
            # Stops updating distribution after seeing this many instances
            self.fix_after = fix_after

        # Fixed version of algorithm - build full distribution now
        else:
            fix_after = fix_after.to('cpu')
            # Fixed from now on
            self.fix_after = -1
            # Build full distribution from the targets provided
            bin_idxs = (self.bin_boundaries.repeat(len(fix_after), 1) <= fix_after.reshape(-1, 1)).sum(-1) - 1
            self.increment_bin_logging(fix_after, bin_idxs)

    def __getattr__(self, name):
        try:
            return getattr(self.loss, name)
        except:
            return super().__getattr__(name)

    def increment_bin_logging(self, transformed_target, bin_idxs):

        # For each of the indices...
        for i in bin_idxs:

            # Increment the raw count of instances found in that bin
            # We cannot parallelise this using self.bin_counts[bin_idxs] += 1 because of repeats
            self.bin_counts[i] += 1

        # Increase total count by batch size
        self.total_count += len(transformed_target)

    def get_weight_vector(self, bin_idxs):

        # Get the number of instances we've seen of these 
        # (including the current ones, since we've already run increment_bin_logging on them)
        bin_populations = self.bin_counts[bin_idxs]

        # Get the weights for each bin, which is just 1/proportion of instances seen in that bin
        weight_vector = 1 - (bin_populations / self.total_count)

        return weight_vector

    def _transform_target(self, target):
        return self.loss._transform_target(target)

    def forward(self, input, target):
        # Transform the data, e.g. make it [0, 1] scale
        transformed_target = self._transform_target(target)

        # Reshape the input to make sure loss works
        input = input.reshape(transformed_target.shape)

        # For reuse later
        bin_idxs = (self.bin_boundaries.repeat(len(transformed_target), 1) < transformed_target.reshape(-1, 1)).sum(-1) - 1

        # Get the BCE loss with the transformed targets, 
        raw_loss = super(AdditionImportanceWeightedBCELoss, self).forward(input, transformed_target)

        if self.total_count <= self.fix_after:
            # If we are still changing the bin weights, increment this
            # Otherwise, we've seen enough to understand full proportion, we don't need to change weightings
            self.increment_bin_logging(transformed_target, bin_idxs)

        # Get the weight vector, of same size as the batch, based on the bins that each instance
        # in the batch falls in
        weight_vector = self.get_weight_vector(bin_idxs)

        # Reweight the losses for each instance in the batch
        weighted_loss = weight_vector @ raw_loss

        return weighted_loss
        

class ImportanceWeightedBCELoss(AdditionImportanceWeightedBCELoss):

    def get_weight_vector(self, bin_idxs):

        # Get the number of instances we've seen of these 
        # (including the current ones, since we've already run increment_bin_logging on them)
        bin_populations = self.bin_counts[bin_idxs]

        # Sticking with IW theory now
        weight_vector = self.total_count / bin_populations

        return weight_vector


if __name__ == '__main__':

    lc_target = torch.rand(50000) - 1
    loss = NonParametricJacobianTransformedBCELoss(lc_target)

    lc_target = torch.linspace(-1, -1/100, 10000)
    transformed_target = loss._transform_target(lc_target)

    import pdb; pdb.set_trace()
