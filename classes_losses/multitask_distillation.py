import os
from numpy import e
from torch import nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence


scheduler_functions = [
    'no_lambda_scheduling',
    'step_pretrainer_lambda_scheduling',
    'asymptotic_pretrainer_lambda_scheduling'
]



## Some of the lambda scheduling functions - see 
## MultiTaskSymmetricTemperatureKLDistillationLoss documentation below
def no_lambda_scheduling(parameter, epoch):
    """
        Flat lambda all the way through training
        parameter = lambda, flat
    """
    return parameter


def step_pretrainer_lambda_scheduling(parameter, epoch):
    """
        Pretrain on just distilltion for some epochs, then jump to just acquisition prediction
        Epoch count starts at 0 => we use < for comparison
        parameter = epoch on which we switch
    """
    return 0. if epoch < parameter else 1.


def asymptotic_pretrainer_lambda_scheduling(parameter, epoch):
    """
        Start with just distillation for the first epoch, the slowly (exponentially)
        schedule the parameter towards 1, i.e. just acquisition prediction
        parameter = time constant in epochs, i.e. ~ the epoch at which lambda = 0.63
    """
    return 1 - e ** -(epoch / parameter)


class SymmetricTemperatureKLDistillationLoss(nn.Module):
    """
        1. Apply temperature annealing to both input and target logits
        2. Turn both into distributions
        3. Get KL(input||target) between each pair
    """
    def __init__(self, temperature):
        self.temperature = temperature
        super(SymmetricTemperatureKLDistillationLoss, self).__init__()

    def forward(self, input, target):
        # Both inputs are logits of size (N, C)

        # Anneal temperatures for both logits, and make distributions
        annealed_input = Categorical(logits = input/self.temperature)
        annealed_target = Categorical(logits = target/self.temperature)
        
        # Return the KL divergence
        # (annealed_input.probs * torch.log(annealed_input.probs/annealed_target.probs)).sum(-1)
        kl_loss = kl_divergence(annealed_input, annealed_target) * (self.temperature**2)
        return kl_loss.mean()


class BaseMultitaskDistillationLoss(nn.Module):

    """
        Base class for multitask (2 loss) distillation losses
        See MultiTaskSymmetricTemperatureKLDistillationLoss for how to build on top of it
    """

    def __init__(self, acquisition_criterion, acquisition_target_weight_parameter, lambda_scheduler, steps_in_epoch, logging_file, **information_rich_criterion_kwargs):

        super(BaseMultitaskDistillationLoss, self).__init__()

        # Initialise the information rich loss function
        self.information_rich_criterion = self.information_rich_criterion_class(**information_rich_criterion_kwargs)

        # Add the secondary criterion for the acquisition head
        self.acquisition_criterion = acquisition_criterion

        # Add the weighting which acquisition_criterion (the acquisition based one) is given
        # This is actually a parameter for lambda_scheduler, see below
        self.acquisition_target_weight_parameter = acquisition_target_weight_parameter

        # This is a function like the ones in this file, which takes a scale parameter and an epoch number
        # and this parameter and returns a lambda value for that whole epoch
        self.lambda_scheduler = lambda_scheduler

        # Steps in epoch lets the loss function know how many steps before it increases self.epoch_num
        print('WARNING: Remember to include length of test dataloader in steps_in_epoch')
        self.steps_in_epoch = steps_in_epoch

        self.t = 0          # Training/testing step we are on for this epoch
        self.epoch_num = 0  # Epoch we are on

        # If we want to log, write the column headers
        self.log_file = logging_file
        if os.path.exists(logging_file):
            raise ValueError(f'{logging_file} already exists!')
        if self.log_file != None:
            self.log("Training", "Lambda", "Distillation", "Secondary", mode='w')

    def log(self, *items, mode='a'):
        if self.log_file != None:
            # Write to log with a comma in the middle
            with open(self.log_file, mode) as f:
                f.write(",".join(list(map(str, items))) + "\n")

    def acquisition_target_weight(self):
        # Pass the function onto the correct scheduling function
        _lambda = self.lambda_scheduler(
            parameter = self.acquisition_target_weight_parameter,
            epoch = self.epoch_num
        )
        assert 0 <= _lambda <= 1
        return _lambda

    def weighted_sum(self, distillation_loss, acquisition_loss, _lambda):
        # Get the scheduled lambda weighting        
        # Return the weighted loss of the two values
        return _lambda * acquisition_loss + (1 - _lambda) * distillation_loss

    def forward(self, input, target):

        # Seperate the input and target out to their right places:
        distillation_input, acquisition_input = input
        distillation_target, acquisition_target = target

        # Get the baseline distillation loss
        distillation_loss = self.information_rich_criterion(distillation_input, distillation_target)

        # Get the secondary, acqusition based loss
        acquisition_loss = self.acquisition_criterion(acquisition_input, acquisition_target)

        # Get the lambda parameter
        weight = self.acquisition_target_weight()

        # log these loss values
        self.log(self.training, weight, distillation_loss.item(), acquisition_loss.item())

        # Before we finish, update the tickers that help parameterise lambda
        self.t += 1

        # If we have hit a new epoch, restart the step counter and increment the epoch counter
        if self.t == self.steps_in_epoch:
            self.t = 0
            self.epoch_num += 1

        # Return the weighted sum of the two
        return self.weighted_sum(distillation_loss, acquisition_loss, weight)


class MultiTaskSymmetricTemperatureKLDistillationLoss(BaseMultitaskDistillationLoss):
    """
        Now, we have two inputs and two targets, in a tuple/list
            First one is related to posterior distillation, which we use SymmetricTemperatureKLDistillationLoss for
            Second one is related to the target acquisition function, for which we do a seperate loss

        To save us a bunch of implementation, we use a logging file directly here to save the history of
        the distillation and secondary losses

        forward is same as base class => no need to redefine
    """

    information_rich_criterion_class = SymmetricTemperatureKLDistillationLoss


class MultiTaskBagOfPhonemesLoss(BaseMultitaskDistillationLoss):
    """
        We have an input dictionary, as in the speech library, from which we must get the information

        The second loss is still for the target acquisition function, i.e. confidence based on
        The first loss is for bag of words/phonemes/trigrams - for which we just use tempered KL divergence like before
    """

    information_rich_criterion_class = SymmetricTemperatureKLDistillationLoss
    information_key = 'phonemes'


    def forward(self, input, target):
        
        ## Need to covert to tuple form so we can pass to base class
        information_key = self.information_key

        # Input does need any change
        # Target is the bag of words and the confidence
        target = (target[information_key], target['confidence'])
        return super(MultiTaskBagOfWordsLoss, self).forward(input, target)


class MultiTaskBagOfWordsLoss(BaseMultitaskDistillationLoss):
    information_key = 'words'


class MultiTaskBagOfTrigramsLoss(BaseMultitaskDistillationLoss):
    information_key = 'trigrams'
