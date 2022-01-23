from torch import nn
from classes_utils.audio.las.attention import SelfAttention, SelfAttentionTranformerLayer
from classes_utils.audio.las.pyramidal_network import pBLSTMLayer, PrepForPyramid
from classes_utils.audio.las.tdnn import MovingDNNLayer, TDNNLayer, TDNNPadding
from util_functions.base import load_state_dict
from classes_utils.layers import BidirectionalLSTMHiddenStateStacker, BidirectionalLSTMOutputSelector, MeanLayer
from classes_architectures.base import AudioEncoderBase, EncoderBase


class SimpleFeedForwardNNEncoder(EncoderBase):
    def __init__(self, mfcc_dim, hidden_dims, embedding_dim, non_lin_func, dropout_rate, variational, num_frames, stride, *args, weights_path=None, **kwargs):
        
        # DEfine the single moving DNN layer
        layers = nn.ModuleList([
            MovingDNNLayer(
                mfcc_dim, num_frames, stride, 
                layer_dims = hidden_dims + [embedding_dim], 
                non_lin_func = non_lin_func, dropout = dropout_rate
            )
        ])
        
        dropout_idxs = []
        noise_idx = None

        super(SimpleFeedForwardNNEncoder, self).__init__(layers, dropout_idxs, noise_idx, variational, embedding_dim, return_skips=False)
        load_state_dict(self, weights_path)


class BidirectionalLSTMAudioEncoder(AudioEncoderBase):

    def __init__(self, mfcc_dim, lstm_sizes, lstm_layers, fc_hidden_dims, embedding_dim, dropout_rate, cell_type, variational, *args, weights_path=None, **kwargs):

        cell_class = {'lstm': nn.LSTM, 'gru': nn.GRU}[cell_type]

        # LSTM layer for mfcc -> lstm_sizes/lstm_layers
        layers = nn.ModuleList([
            cell_class(input_size=mfcc_dim, hidden_size=lstm_sizes[0]//2, num_layers=lstm_layers[0], 
                bidirectional=True, batch_first=True, dropout=dropout_rate),
            BidirectionalLSTMOutputSelector(cell_type)
        ])
        dropout_idxs = []
        # String of LSTM layers
        for i, lstm_layer in enumerate(lstm_layers[1:]):
            layers.extend([
                cell_class(input_size=lstm_sizes[i], hidden_size=lstm_sizes[i+1]//2, num_layers=lstm_layer,
                    bidirectional=True, batch_first=True, dropout=dropout_rate),
                # Feed final LSTM through BidirectionalLSTMHiddenStateStacker instead
                BidirectionalLSTMOutputSelector(cell_type) if i != len(lstm_layers) - 2 else BidirectionalLSTMHiddenStateStacker(cell_type)
            ])
            dropout_idxs.append(len(layers)-2)
        noise_idx = len(layers)
        if len(fc_hidden_dims) > 0:
            # First FC layer
            layers.extend([
                # Stacking by BidirectionalLSTMHiddenStateStacker
                nn.Linear(lstm_layers[-1]*lstm_sizes[-1], fc_hidden_dims[0]),
                nn.Dropout(p=dropout_rate),
                nn.Sigmoid(),
            ])
            dropout_idxs.append(len(layers)-2)
            # String of FC layers
            for i, hd in enumerate(fc_hidden_dims[:-1]):
                layers.append(nn.Linear(hd, fc_hidden_dims[i+1]))
                layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.Sigmoid())
                dropout_idxs.append(dropout_idxs[-1]+2)

            layers.append(nn.Linear(fc_hidden_dims[-1], embedding_dim))

        else:
            layers.append(nn.Linear(lstm_layers[-1]*lstm_sizes[-1], embedding_dim))

        super(BidirectionalLSTMAudioEncoder, self).__init__(mfcc_dim, layers, dropout_idxs, noise_idx, variational, embedding_dim)
        load_state_dict(self, weights_path)


class BidirectionalLSTMOnlyAudioEncoder(AudioEncoderBase):

    def __init__(self, mfcc_dim, lstm_sizes, lstm_layers, embedding_dim, dropout_rate, cell_type, variational, *args, weights_path=None, **kwargs):

        cell_class = {'lstm': nn.LSTM, 'gru': nn.GRU}[cell_type]

        # LSTM layer for mfcc -> lstm_sizes/lstm_layers
        layers = nn.ModuleList([
            cell_class(input_size=mfcc_dim, hidden_size=lstm_sizes[0]//2, num_layers=lstm_layers[0], 
            bidirectional=True, batch_first=True, dropout=dropout_rate),
            BidirectionalLSTMOutputSelector(cell_type)
        ])
        # String of LSTM layers
        for i, lstm_layer in enumerate(lstm_layers[1:]):
            # Bidirectional => output is double size that we put in
            layers.extend([
                cell_class(input_size=lstm_sizes[i], hidden_size=lstm_sizes[i+1]//2, num_layers=lstm_layer,
                    bidirectional=True, batch_first=True, dropout=dropout_rate),
                BidirectionalLSTMOutputSelector(cell_type)
            ])
        noise_idx = len(layers)
        # May need to address this
        dropout_idxs = []
        # First FC layer
        layers.extend([
            cell_class(input_size=lstm_sizes[-1], hidden_size=embedding_dim//2, num_layers=lstm_layer,
                bidirectional=True, batch_first=True, dropout=dropout_rate),
            BidirectionalLSTMOutputSelector(cell_type)
        ])

        super(BidirectionalLSTMOnlyAudioEncoder, self).__init__(mfcc_dim, layers, dropout_idxs, noise_idx, variational, embedding_dim)
        load_state_dict(self, weights_path)


class BLSTMListenerSelfAttentionEncoder(AudioEncoderBase):
    
    def __init__(self, mfcc_dim, lstm_hidden_size, pyramid_size, key_size, query_size, value_size, num_heads, dropout, variational):

        layers = [PrepForPyramid(pyramid_size=pyramid_size)]
        layers += [pBLSTMLayer(input_size=mfcc_dim, hidden_size=lstm_hidden_size, num_layers=1, dropout=dropout)]
        layers += [pBLSTMLayer(input_size=lstm_hidden_size*2, hidden_size=lstm_hidden_size, num_layers=1, dropout=dropout)] * (pyramid_size-1)
        layers += [SelfAttention(lstm_hidden_size*2, num_heads, query_size, key_size, value_size)]
        layers += [MeanLayer(dim=1)]
        
        layers = nn.ModuleList(layers)

        super(BLSTMListenerSelfAttentionEncoder, self).__init__(mfcc_dim, layers, [], 0, variational, value_size*num_heads)


class BLSTMListenerTransformerEncoder(AudioEncoderBase):
    
    def __init__(self, mfcc_dim, pyramid_size, d_model, num_heads, hidden_sizes, num_attn_blocks, dropout, variational):

        layers = [PrepForPyramid(pyramid_size=pyramid_size)]
        layers += [pBLSTMLayer(input_size=mfcc_dim, hidden_size=d_model//2, num_layers=1, dropout=dropout)]
        layers += [pBLSTMLayer(input_size=d_model, hidden_size=d_model//2, num_layers=1, dropout=dropout)] * (pyramid_size-1)
        layers += [SelfAttentionTranformerLayer(d_model=d_model, num_heads=num_heads, hidden_sizes=hidden_sizes)] * num_attn_blocks
        layers += [MeanLayer(dim=1)]
        
        layers = nn.ModuleList(layers)

        super(BLSTMListenerTransformerEncoder, self).__init__(mfcc_dim, layers, [], 0, variational, d_model)


class SlidingDNNListenerSelfAttentionEncoder(AudioEncoderBase):

    def __init__(
        self, mfcc_dim, listener_num_frames, listener_strides, listener_hidden_dims, listener_non_lin_funcs,
        key_size, query_size, value_size, num_heads, dropout, variational
    ):
        layers = [
            MovingDNNLayer(mfcc_dim, listener_num_frames[0], listener_strides[0], 
            listener_hidden_dims[0], listener_non_lin_funcs[0], dropout)
        ]
        for i in range(1, len(listener_hidden_dims)):
            layers.append(
                MovingDNNLayer(listener_hidden_dims[i-1][-1], listener_num_frames[i], listener_strides[i], 
                listener_hidden_dims[i], listener_non_lin_funcs[i], dropout)
            )
        layers += [SelfAttention(listener_hidden_dims[-1][-1], num_heads, query_size, key_size, value_size)]
        layers += [MeanLayer(dim=1)]

        layers = nn.ModuleList(layers)

        super(SlidingDNNListenerSelfAttentionEncoder, self).__init__(mfcc_dim, layers, [], 0, variational, value_size*num_heads)


class SlidingDNNListenerTransformerEncoder(AudioEncoderBase):

    def __init__(
        self, mfcc_dim, listener_num_frames, listener_strides, listener_hidden_dims, listener_non_lin_funcs, 
        d_model, num_heads, hidden_sizes, num_attn_blocks, dropout, variational
    ):
        layers = [
            MovingDNNLayer(mfcc_dim, listener_num_frames[0], listener_strides[0], 
            listener_hidden_dims[0], listener_non_lin_funcs[0], dropout)
        ]
        for i in range(1, len(listener_hidden_dims)):
            layers.append(
                MovingDNNLayer(listener_hidden_dims[i-1][-1], listener_num_frames[i], listener_strides[i], 
                listener_hidden_dims[i], listener_non_lin_funcs[i], dropout)
            )
        layers += [SelfAttentionTranformerLayer(d_model=d_model, num_heads=num_heads, hidden_sizes=hidden_sizes)] * num_attn_blocks
        layers += [MeanLayer(dim=1)]

        layers = nn.ModuleList(layers)

        super(SlidingDNNListenerTransformerEncoder, self).__init__(mfcc_dim, layers, [], 0, variational, d_model)


class TDNNListenerSelfAttentionEncoder(AudioEncoderBase):
    
    def __init__(
        self, mfcc_dim, listener_output_dims, listener_context_idxs, listener_non_lin_funcs, listener_strides, 
        key_size, query_size, value_size, num_heads, dropout, variational
    ):
        tdnn_layers = [
            TDNNLayer(mfcc_dim, listener_output_dims[0], listener_context_idxs[0], 
            listener_non_lin_funcs[0], listener_strides[0],  dropout)
        ]
        for i in range(1, len(listener_output_dims)):
            tdnn_layers.append(
                TDNNLayer(listener_output_dims[i-1], listener_output_dims[i], listener_context_idxs[i], 
                listener_non_lin_funcs[i], listener_strides[i],  dropout)
            )
        layers = [TDNNPadding(tdnn_layers)] + tdnn_layers
        layers += [SelfAttention(listener_output_dims[-1], num_heads, query_size, key_size, value_size)]
        layers += [MeanLayer(dim=1)]

        layers = nn.ModuleList(layers)

        super(TDNNListenerSelfAttentionEncoder, self).__init__(mfcc_dim, layers, [], 0, variational, value_size*num_heads)


class TDNNListenerTransformerEncoder(AudioEncoderBase):
    
    def __init__(
        self, mfcc_dim, listener_output_dims, listener_context_idxs, listener_non_lin_funcs, listener_strides, 
        d_model, num_heads, hidden_sizes, num_attn_blocks, dropout, variational
    ):
        tdnn_layers = [
            TDNNLayer(mfcc_dim, listener_output_dims[0], listener_context_idxs[0], 
            listener_non_lin_funcs[0], listener_strides[0],  dropout)
        ]
        for i in range(1, len(listener_output_dims)):
            tdnn_layers.append(
                TDNNLayer(listener_output_dims[i-1], listener_output_dims[i], listener_context_idxs[i], 
                listener_non_lin_funcs[i], listener_strides[i],  dropout)
            )
        layers = [TDNNPadding(tdnn_layers)] + tdnn_layers
        layers += [SelfAttentionTranformerLayer(d_model=d_model, num_heads=num_heads, hidden_sizes=hidden_sizes)] * num_attn_blocks
        layers += [MeanLayer(dim=1)]

        layers = nn.ModuleList(layers)

        super(TDNNListenerTransformerEncoder, self).__init__(mfcc_dim, layers, [], 0, variational, d_model)



if __name__ == '__main__':
    encoder = None