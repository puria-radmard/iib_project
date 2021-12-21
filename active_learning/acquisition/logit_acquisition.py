from active_learning.util_classes import RenovationError
from .utils import *

class RandomBaselineAcquisition(UnitwiseAcquisition):
    def __init__(self, dataset, score_shape_func):
        super().__init__(dataset=dataset)
        # i.e. a function that takes in a single datapoint and outputs the shape of the score
        # requires both dataset and i to be input
        self.score_shape_func = score_shape_func
        # self.score_shape = self.dataset.last_logits[0].max(axis=-1).shape

    def score(self, batch_indices):
        batch_output = []
        for i in batch_indices:
            # Add an axis when batching
            score_shape = self.score_shape_func(self.dataset, i)
            batch_output.append(torch.randn(score_shape))
        return batch_output


class LowestConfidenceAcquisition(UnitwiseAcquisition):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, batch_indices):
        batch_output = []
        for i in batch_indices:
            batch_output.append(- self.dataset.last_logits[i].max(axis=-1).values)
        return batch_output


class MarginAcquisition(UnitwiseAcquisition):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, batch_indices):
        batch_output = []
        for i in batch_indices:
            logits = self.dataset.last_logits[i]
            batch_output.append(- (logits.max(axis=-1).values - logits.min(axis=-1).values))
        return batch_output


class MaximumEntropyAcquisition(UnitwiseAcquisition):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, batch_indices):
        batch_output = []
        for i in batch_indices:
            cater = torch.distributions.Categorical(logits=self.dataset.last_logits[i])
            batch_output.append(cater.entropy())
        return batch_output


class PredsKLAcquisition(DataAwareAcquisition):
    def __init__(self, dataset):
        super().__init__(dataset=dataset)

    def score(self, batch_indices):
        batch_output = []
        for i in batch_indices:
            p_t1 = torch.distributions.Categorical(logits=self.dataset.last_logits.prev_attr[i])
            p_t2 = torch.distributions.Categorical(logits=self.dataset.last_logits[i])
            forward_kl = torch.distributions.kl.kl_divergence(p_t1, p_t2)
            backward_kl = torch.distributions.kl.kl_divergence(p_t2, p_t1)
            batch_output.append(0.5*(forward_kl + backward_kl))
        return batch_output

    def step(self):
        pass


class EmbeddingMigrationAcquisition(DataAwareAcquisition):
    def __init__(self, dataset, embedding_name):
        super().__init__(dataset=dataset)
        self.embedding_name = embedding_name

    def score(self, batch_indices):
        batch_output = []
        for i in batch_indices:
            embs = self.dataset.__getattr__(self.embedding_name)[i]
            previous_embs = self.dataset.__getattr__(self.embedding_name).prev_attr[i]
            batch_output.append(torch.cdist(embs, previous_embs))
        return batch_output

    def step(self):
        pass
