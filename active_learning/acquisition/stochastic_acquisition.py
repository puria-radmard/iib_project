from .utils import *

class VAEEnsembleKnowledgeUncertainty(UnitwiseAcquisition):
    def __init__(self, dataset):
        super(VAEEnsembleKnowledgeUncertainty, self).__init__(dataset=dataset)
        assert "vae_ensemble_gaussians" in dataset.attrs and "vae_ensemble_embeddings" in dataset.attrs

    def score(self, i):
        raise NotImplementedError
        return self.dataset.vae_ensemble_embeddings.instance_entropy(i) - self.dataset.vae_ensemble_embeddings.instance_entropy(i)


class BALDAcquisition(UnitwiseAcquisition):
    def __init__(self, dataset):
        super(BALDAcquisition, self).__init__(dataset)

    def score(self, batch_indices):
        # batch x ensemble_size x num_classes
        logits_list = self.dataset.last_logits[batch_indices]
        distributions = torch.distributions.Categorical(logits = logits_list)

        # Total = entropy of mean
        mean_probs = distributions.probs.mean(1)
        total_uncertainty = torch.distributions.Categorical(probs = mean_probs).entropy()

        # data = mean of entropies
        data_uncertainty = distributions.entropy().mean(-1)

        return total_uncertainty - data_uncertainty
