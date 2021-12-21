import torch
from torch import nn
from util_functions.base import length_aware_loss
device = (
    torch.device('cuda') if torch.cuda.is_available() 
    else torch.device('cpu')
)

__all__ = [
    'ReconstructionLoss'
]

class ReconstructionLoss(nn.Module):
    def __init__(self, mean_decodings, mean_losses=True):
        super(ReconstructionLoss, self).__init__()
        self.mean_decodings = mean_decodings
        # reduction = 'mean' if mean_losses else 'none'
        self.criterion = torch.nn.MSELoss(reduction='none').to(device)
        self.mean_losses = mean_losses

    def forward(self, decodings, batch):
        if self.mean_decodings:
            decodings = [torch.mean(torch.stack(decodings), dim=0)]
        gt = batch['padded_features'].to(device)
        lengths = batch['lengths']
        losses = [self.criterion(d, gt) for d in decodings]
        crits_and_masks = [length_aware_loss(loss_sample, lengths) for loss_sample in losses]
        if self.mean_losses:
            return torch.mean(torch.stack([co.sum()/lm.sum() for co, lm in crits_and_masks]))
        else:
            return [cam[0] for cam in crits_and_masks], crits_and_masks[0][1] # loss masks should all be the same


class ImageReconstructionLoss(ReconstructionLoss):

    def forward(self, reconstructions, inputs):
        if self.mean_decodings:
            reconstructions = [torch.mean(torch.stack(reconstructions), dim=0)]
        losses = [self.criterion(d, inputs).mean() for d in reconstructions]
        return torch.mean(torch.stack(losses)) if self.mean_losses else losses



class MovingReconstructionLoss(nn.Module):
    def __init__(self, num_frames, stride, mean_decodings, mean_losses=True):
        super(MovingReconstructionLoss, self).__init__()
        self.mean_decodings = mean_decodings
        # reduction = 'mean' if mean_losses else 'none'
        self.criterion = torch.nn.MSELoss(reduction='none').to(device)
        self.mean_losses = mean_losses
        self.num_frames = num_frames
        self.stride = stride

    @staticmethod
    def get_loss_mask(gt, lengths):
        loss_mask = torch.ones_like(gt)
        for j, lm in enumerate(loss_mask):
            lm[lengths[j]:] = 0
        return loss_mask

    def slide_over(self, tensor):
        output = []
        [batch, seqlen, mfcc] = tensor.shape
        for i in list(range(0, seqlen, self.stride))[:-1]:
            in_frames = tensor[:,i:i+self.num_frames]
            if in_frames.shape[1] == self.num_frames:
                output.append(in_frames)

        return torch.stack(output, 1)

    def forward(self, decodings, batch):
        if self.mean_decodings:
            decodings = [torch.mean(torch.stack(decodings), dim=0)]
        lengths = batch['lengths']
        batch_gt = batch['padded_features'].to(device)
        lm = self.get_loss_mask(batch_gt, lengths)
        slide_gt = self.slide_over(batch_gt)
        slide_lm = self.slide_over(lm)
        losses = [self.criterion(d, slide_gt) for d in decodings]
        crits_and_masks = [(losses[i]*slide_lm, slide_lm) for i in range(len(losses))]
        if self.mean_losses:
            return torch.mean(torch.stack([co.sum()/lm.sum() for co, lm in crits_and_masks]))
        else:
            return [cam[0] for cam in crits_and_masks], crits_and_masks[0][1] # loss masks should all be the same
