from classes_architectures.audio.encoder import *
from classes_architectures.audio.decoder import *
from classes_architectures.cifar.encoder import *
from classes_architectures.cifar.decoder import *
from classes_utils.layers import *

try:
    from cifar_repo.cifar import make_model
except:
    def make_model(*args, **kwargs):
        raise Exception('cifar library not found')

## FOR BACKWARDS REFERENCE ONLY
# moving_encoder_types = {
#     "simple_sliding_nn": SimpleFeedForwardNNEncoder
# }
# encoder_types = moving_encoder_types.copy()
# encoder_types.update({
#     "basic_bidirectional_LSTM": BidirectionalLSTMAudioEncoder,
#     "only_bidirectional_LSTM": BidirectionalLSTMOnlyAudioEncoder,
#     "unet": UNetEncoder,
#     "no_skip": NoSkipEncoder,
#     "pytorch-classification": make_model,
#     "blstm_listener_self_attention": BLSTMListenerSelfAttentionEncoder,
#     "blstm_listener_transformer": BLSTMListenerTransformerEncoder,
#     "simple_loader": EmbeddingLoaderEncoder
# })
# 
# moving_decoder_types = {
#     "simple_sliding_nn": SimpleFeedForwardNNDecoder,
# }
# decoder_types = moving_decoder_types.copy()
# decoder_types.update({
#     "no_decoder": EmptyLayer, 
#     "basic_LSTM": BasicLSTMAudioDecoder,
#     "autoregressive_LSTM": AutoRegressiveLSTMAudioDecoder,
#     "fc_decoder": FCDecoder,
#     "unet": UNetDecoder,
#     "no_skip": NoSkipDecoder,
#     "staircase": StaircaseConvolutionalDecoder
# })
