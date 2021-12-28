import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf
from nlpaug.util import Logger

import numpy as np
import torch


class FixedFrequencyMaskingAug(nas.FrequencyMaskingAug):
    """
    Original had some issues that I needed to fix
    """
    def __init__(self, name='FrequencyMasking_Aug', zone=(0.2, 0.8), coverage=1., factor=(40, 80), verbose=0, 
        silence=False, stateless=True):
        super().__init__(zone=zone, coverage=coverage, factor=factor, verbose=verbose, 
            name=name, silence=silence, stateless=stateless)

    def manipulate(self, data, f, f0, time_start, time_end):
        # Defined here instead of model
        aug_data = torch.clone(data)
        aug_data[f0:f0+f, time_start:time_end] = 0
        return aug_data

    def substitute(self, data):
            v = data.shape[0]
            if v < self.factor[1] and not self.silence:
                Logger.log().warning('Upper bound of factor is larger than {}.'.format(v) + 
                ' It should be smaller than number of frequency. Will use {} as upper bound'.format(v))

            upper_bound = self.factor[1] if v > self.factor[1] else v
            
            # Changed this to make lower bound respected
            # f = self.get_random_factor(high=upper_bound, dtype='int')
            # f0 = np.random.randint(v - f)
            f1 = self.get_random_factor(high=upper_bound, dtype='int') + 1
            f0 = np.random.randint(self.factor[0], f1)
            f = f1 - f0
            
            # Changed here to make sure frequency mask goes across full time range
            # time_start, time_end = self.get_augment_range_by_coverage(data)
            time_start, time_end = self.get_augment_range_by_coverage(data[0])

            if not self.stateless:
                self.v, self.f, self.f0, self.time_start, self.time_end = v, f, f0, time_start, time_end

            # To allow use with tensors
            # return self.model.manipulate(data, f=f, f0=f0, time_start=time_start, time_end=time_end)
            return self.manipulate(data, f=f, f0=f0, time_start=time_start, time_end=time_end)


class TransformationDistribution:
    """
        Takes only transform_hyperparameter_dict, which is of form(s):
            frequency: bands and coverages that can be augmented randomly, i.e. [(f1, f2, c1), (f3, f4, c2), ...]
        THIS FEATURE IS NOT USED YET!
        Requirements:
            - Frquency band size << mfcc_dim
    """
    
    def __init__(self, transform_hyperparameter_dict, mfcc_dim):
        self.mfcc_dim = mfcc_dim
        aug_list = []
        if "frequency" in transform_hyperparameter_dict:
            for band in transform_hyperparameter_dict['frequency']:
                assert band[1] < mfcc_dim and all([b >= 0 for b in band]) and len(band) == 3 and band[-1]<=1
                aug_list.append(
                    FixedFrequencyMaskingAug(factor = (band[0], band[1]), zone = (0, 1), coverage = band[2])
                )
        self.flow = naf.Sequential(aug_list)

    def __call__(self, instance):
        assert len(instance.shape) == 2, "Not implemented for batches yet - must pass 2 dimensional array to this transformation"
        if instance.shape[0] == self.mfcc_dim:
            data = instance
        elif instance.shape[1] == self.mfcc_dim:
            data = instance.T
        return self.flow.augment(data)