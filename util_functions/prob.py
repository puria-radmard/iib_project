import torch
from util_functions.base import batch_trace, batch_outer


def fit_gaussian(embedding_samples, embedding_means):
    sample_covariances = sum(
        batch_outer(es, embedding_means) for es in embedding_samples
    ) / (len(embedding_samples) - 1)
    return sample_covariances


def gaussian_KL(cov1, cov2, mean_diff):
    inv2 = torch.inverse(cov2)
    logd_term = torch.logdet(cov2) - torch.logdet(cov1)
    trce_term = batch_trace(inv2 @ cov1)
    quad_term = mean_diff.T @ (inv2 @ mean_diff)
    d = mean_diff.size(-1)
    return 0.5 * (logd_term - d + trce_term + quad_term)


def mean_NLL(distribution, point_samples):
    return torch.sum(-distribution.log_prob(ps) for ps in point_samples) / len(
        point_samples
    )


def reparameterised_draw(mean, log_var):
    std = torch.exp(log_var * 0.5)
    eps = torch.randn_like(std)
    return mean + eps * std
