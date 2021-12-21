from .utils import *


class ReconstructionLossOfAverageMetric(UnitwiseAcquisition):
    """
    L_1
    If self.dataset is stochastic (C > 1):
        this takes the average of each reconstruction and takes the MSE loss of that compared to target
    Else (C = 1):
        throws error
    NB: Meaned over pixels => should be of same size
    """

    def __init__(self, dataset):
        assert dataset.is_stochastic, 'ReconstructionLossOfAverageMetric requires stochastic dataset'
        super().__init__(dataset)    

    def score(self, batch_indices):
        decodings = [self.dataset.last_logits[i] for i in batch_indices]
        targets, _, lengths, _ = self.dataset.get_batch(batch_indices, False)
        targets = targets.to(device)

        # decodings is a list of size batch_indices
        # each list in decodings is of size C (num ensemble samples)
        # for each item in decodings, average it over the ensemble size (i.e. average reconstruction)
        average_reconstructions = [d.mean(dim=0) for d in decodings]
        
        # take mse loss for each image:
        metric_tensor = F.mse_loss(torch.stack(average_reconstructions), targets, reduction='none')
        
        # For sequences, this is of size [batch, 1, seq_len, feature_size]
        # For images, lengths is just Nones anyway so it's fine!
        image_metrics = []
        for item_metric, length in zip(metric_tensor, lengths):
            image_metrics.append(
                item_metric.reshape(-1, item_metric.shape[-2])[:length].mean()
            )

        return image_metrics
        

class AverageOfReconstructionLossesMetric(UnitwiseAcquisition):
    """
    L_2
    If self.dataset is stochastic (C > 1):
        this takes the MSE loss of each reconstruction and averages them
    Else (C = 1):
        this takes the MSE loss of the single reconstructions
    NB: Meaned over pixels => should be of same size
    """

    def __init__(self, dataset):
        assert dataset.is_stochastic, 'AverageOfReconstructionLossesMetric requires stochastic dataset'
        super().__init__(dataset)

    def score(self, batch_indices):
        decodings = [self.dataset.last_logits[i] for i in batch_indices]
        targets, _, lengths, _ = self.dataset.get_batch(batch_indices, False)
        targets = targets.to(device)
        
        # i.e. list of list of samples
        num_samples = self.dataset.last_logits.C
        mse_losses = []
        for c in range(num_samples):
            # For cth ensemble index, take cth reconstruction of each image
            sample = torch.stack([d[c] for d in decodings])
            # Compare this to targets
            # This is of shape 
            metric_sample = F.mse_loss(sample, targets, reduction='none')
            mse_losses.append(metric_sample)

        # Average loss of samples
        average_mse = torch.mean(torch.stack(mse_losses), 0)

        # For sequences, this is of size [batch, 1, seq_len, feature_size]
        # For images, lengths is just Nones anyway so it's fine!
        image_metrics = []
        for item_metric, length in zip(average_mse, lengths):
            image_metrics.append(
                item_metric.reshape(-1, item_metric.shape[-2])[:length].mean()
            )
    
        return image_metrics



class ReconstructionDisagreementMetric(UnitwiseAcquisition):
    """
    L_3 = L_2 - L_1
    If self.dataset is stochastic (C > 1):
        average MSE between members of ensemble and the average of the reconstruction
        (not done explicitly)
    Else (C = 1):
        throws error
    NB: Meaned over pixels => should be of same size
    """

    def __init__(self, dataset):
        assert dataset.is_stochastic, 'ReconstructionDisagreementMetric requires stochastic dataset'
        self.l2 = AverageOfReconstructionLossesMetric(dataset)
        self.l1 = ReconstructionLossOfAverageMetric(dataset)
        super().__init__(dataset)    

    def score(self, batch_indices):
        return self.l2.score(batch_indices) - self.l1.score(batch_indices)
