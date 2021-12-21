import sys, torch
from torch import nn
from util_functions.base import load_state_dict
from classes_architectures.base import DecoderBase
from classes_utils.layers import EmptyLayer

__all__ = [
    "BasicLSTMAudioDecoder",
    "FCDecoder",
    "SimpleFeedForwardNNDecoder",
    "AutoRegressiveLSTMAudioDecoder"
]

device = (
    torch.device('cuda') if torch.cuda.is_available() 
    else torch.device('cpu')
)

class SimpleFeedForwardNNDecoder(DecoderBase):
    def __init__(self, mfcc_dim, embedding_dim, hidden_dims, num_frames, mean_first, *args, reverse_hidden_dims = True, weights_path = None, **kwargs):
        super(SimpleFeedForwardNNDecoder, self).__init__(mean_first=mean_first)
        self.mfcc_dim = mfcc_dim
        self.num_frames = num_frames
        if reverse_hidden_dims:
            hidden_dims=hidden_dims[::-1]
        self.layers = nn.ModuleList(
            [
                nn.Linear(embedding_dim, hidden_dims[0]),
                nn.Tanh()
            ]
        )
        for i, hd in enumerate(hidden_dims[:-1]):
            self.layers.append(nn.Linear(hd, hidden_dims[i+1]))
            self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(hidden_dims[-1], mfcc_dim*num_frames))
        load_state_dict(self, weights_path)

    def forward_method(self, x, *args, **kwargs):
        batch_size, *_ = x.shape
        for l in self.layers:
            x = l(x)
        return x.view(batch_size, -1, self.num_frames, self.mfcc_dim)


class BasicLSTMAudioDecoder(DecoderBase):
    def __init__(self, mfcc_dim, embedding_dim, dropout_rate, cell_type, mean_first, *args, weights_path = None, **kwargs):
        super(BasicLSTMAudioDecoder, self).__init__(mean_first=mean_first)
        raise Exception('Unused now')
        self.mfcc_dim = mfcc_dim
        self.embedding_dim = embedding_dim
        self.hidden_size = self.mfcc_dim * 2
        self.decoder = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            dropout=dropout_rate,
            bidirectional=False,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_size, self.mfcc_dim)
        load_state_dict(self, weights_path)

    def forward_method(self, embeddings, seq_len, *args, **kwargs):
        embeddings = embeddings.unsqueeze(1).repeat(1, seq_len, 1)
        out, (_, _) = self.decoder(embeddings)
        rec = self.fc(out)
        return rec


class AutoRegressiveLSTMAudioDecoder(DecoderBase):
    def __init__(self, mfcc_dim, embedding_dim, dropout_rate, cell_type, mean_first, *args, weights_path = None, **kwargs):
        super(AutoRegressiveLSTMAudioDecoder, self).__init__(mean_first=mean_first)
        raise Exception('Integrate gru & customisation support')
        self.mfcc_dim = mfcc_dim
        self.embedding_dim = embedding_dim

        self.decoder = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            bidirectional=False,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(self.embedding_dim, self.mfcc_dim)

    def forward_method(self, embeddings, seq_len, *args, **kwargs):
        # Embeddings initially have size (batch, hidden dim)
        batch_size, _ = embeddings.shape

        # Reshape to add dim 1 to match LSTM input
        embeddings = embeddings.unsqueeze(1)
        # Initialise hidden state, could be zero instead
        hidden = (
            torch.randn(1, batch_size, self.embedding_dim).to(device),
            torch.randn(1, batch_size, self.embedding_dim).to(device)
        )

        # Reconstruction history
        embeddings_history = []
        for i in range(seq_len):
            # Autoregressive
            embeddings, hidden = self.decoder(embeddings, hidden)
            embeddings_history.append(embeddings)

        embeddings = torch.cat(embeddings_history, axis=1)
        embeddings = self.dropout(embeddings)
        recon = self.fc(embeddings)
        return recon


class FCDecoder(DecoderBase):
    def __init__(self, embedding_dim, layer_dims, nonlinearities, dropout_rate, mean_first, *args, weights_path = None, **kwargs):
        super(FCDecoder, self).__init__(mean_first=mean_first)
        
        # nonlinearities[i] is placed after the layer that OUTPUTS layer_dims[i]
        non_lin_dict = {'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'none': EmptyLayer}
        layers = [
            nn.Linear(embedding_dim, layer_dims[0]),
            non_lin_dict[nonlinearities[0]](),
            nn.Dropout(dropout_rate)
        ]
        for i, ld in enumerate(layer_dims[:-1]):
            layers.extend([
                nn.Linear(ld, layer_dims[i+1]),
                non_lin_dict[nonlinearities[i+1]](),
                nn.Dropout(dropout_rate)      
            ])
        # Do not apply dropout to final output layer
        layers = layers[:-1]
        self.layers = nn.Sequential(*layers)

    def forward_method(self, embeddings, *args, **kwargs):
        return self.layers(embeddings)


if __name__ == '__main__':

    from classes_utils.architecture_integration import AudioEncoderDecoderEnsemble

    ensemble_type='basic'
    encoder_type='only_bidirectional_LSTM'
    decoder_type='no_decoder'
    ensemble_size=1
    encoder_ensemble_kwargs=dict(
        mfcc_dim=40, 
        lstm_sizes=[20, 20, 20], 
        lstm_layers=[3, 3, 3],
        embedding_dim=40,
        dropout_rate=0.3, 
        cell_type='gru', 
        variational=False
    )
    decoder_ensemble_kwargs={}

    autoencoder = AudioEncoderDecoderEnsemble(
        ensemble_type=ensemble_type, 
        encoder_type=encoder_type, 
        decoder_type=decoder_type, 
        ensemble_size=ensemble_size, 
        encoder_ensemble_kwargs=encoder_ensemble_kwargs, 
        decoder_ensemble_kwargs=decoder_ensemble_kwargs
    )

    batch = torch.randn(32, 300, 40)

    output = autoencoder(batch)
