import torch
from torch import nn

device = (
    torch.device('cuda') if torch.cuda.is_available() 
    else torch.device('cpu')
)

__all__ = [
    'SimCLREnsemblanceLoss'
]

class SimCLREnsemblanceLoss(nn.Module):
    def __init__(self, temperature, batch_size):
        super(SimCLREnsemblanceLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss().to(device)

    @staticmethod
    def find_positive_indices(batch_size):
        enum_item = enumerate(range(1, batch_size*2, 2), 1)
        index_finder = lambda n, i: batch_size*i-int(0.5*n**2-0.5*n+1)
        positive_indices = torch.tensor([index_finder(n, i) for n, i in enum_item])
        return positive_indices

    def forward_old(self, decodings,  batch):
        # MEANS these to get distance? makes sense to me
        means = torch.mean(torch.stack(decodings), axis = 0)
        batch_size = means.shape[0]//2
        # Need to make this cosine, not euclidean
        pairwise_distances = torch.nn.functional.pdist(means) / self.temperature
        positive_indices = self.find_positive_indices(batch_size)

    def forward(self, decodings, batch):
        # This is lifted directly from https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/simclr.py
        # Please clean this up!
        means = torch.mean(torch.stack(decodings), axis = 0)
        batch_size = means.shape[0]//2
        features = torch.nn.functional.normalize(means)
        similarity_matrix = torch.matmul(features, features.T)
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        logits = logits / self.temperature
        return self.criterion(logits, labels)
