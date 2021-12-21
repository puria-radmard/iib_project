from classes_losses.replication import *

ve_loss_dict = {
    "kl_forward": VAEEnsembleKLReplicationLoss, 
    "kl_reverse": VAEEnsembleReverseKLReplicationLoss,
    "nll_forward": VAEEnsembleNLLReplicationLoss, 
    "nll_forward": VAEEnsembleReverseNLLReplicationLoss
}