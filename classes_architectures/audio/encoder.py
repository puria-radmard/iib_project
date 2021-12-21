import torch
import numpy as np
from torch import nn
from torch.nn.modules.flatten import Flatten
from classes_utils.audio.las.attention import FixedLengthSelfAttention
from classes_utils.audio.las.pyramidal_network import pBLSTMLayer
from util_functions.base import load_state_dict
from classes_utils.layers import BidirectionalLSTMHiddenStateStacker, BidirectionalLSTMOutputSelector
from classes_architectures.base import StaticAudioEncoderBase, MovingAudioEncoderBase


__all__ = [
    'BidirectionalLSTMAudioEncoder',
    'BidirectionalLSTMOnlyAudioEncoder',
    'SimpleFeedForwardNNEncoder'
]


class SimpleFeedForwardNNEncoder(MovingAudioEncoderBase):
    def __init__(self, mfcc_dim, hidden_dims, embedding_dim, dropout_rate, variational, num_frames, stride, *args, weights_path=None, **kwargs):

        # Define simple feed forward network
        self.num_frames = num_frames
        layers = nn.ModuleList([Flatten()])
        # raise Exception('Need non-linearities and dropout here')
        layers.append(nn.Linear(mfcc_dim*num_frames, hidden_dims[0]))
        layers.append(nn.Tanh())
        for i, hd in enumerate(hidden_dims[:-1]):
            layers.append(nn.Linear(hd, hidden_dims[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dims[-1], embedding_dim))
        
        dropout_idxs = []
        noise_idx = None

        super(SimpleFeedForwardNNEncoder, self).__init__(mfcc_dim, layers, dropout_idxs, noise_idx, variational, embedding_dim, stride)
        load_state_dict(self, weights_path)


class BidirectionalLSTMAudioEncoder(StaticAudioEncoderBase):

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


class BidirectionalLSTMOnlyAudioEncoder(StaticAudioEncoderBase):

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


class LASEncoder(StaticAudioEncoderBase):
    
    def __init__(self, mfcc_dim, hidden_size, pyramid_size, key_size, query_size, value_size, variational):

        layers = [pBLSTMLayer(input_size=mfcc_dim, hidden_size=hidden_size, num_layers=1, dropout=0.0)]
        layers += [pBLSTMLayer(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1, dropout=0.0)] * (pyramid_size-1)
        layers += [FixedLengthSelfAttention(mfcc_dim, query_size, key_size, value_size)]

        super(LASEncoder, self).__init__(mfcc_dim, layers, [], 0, variational, value_size)

