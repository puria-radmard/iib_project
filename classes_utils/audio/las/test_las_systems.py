import torch
from torch import nn
from classes_utils.audio.las.attention import FixedLengthSelfAttention
from classes_utils.audio.las.pyramidal_network import pBLSTMLayer, pLSTMOutputExtractor

batch_size = 4
sequence_length = 4000
feature_dim = 40

data = torch.randn(batch_size, sequence_length, feature_dim)

hidden_size = 256

layers = nn.Sequential(
    # Librispeech default config
    pBLSTMLayer(input_size=40, hidden_size=hidden_size, num_layers=1, dropout=0.0),
    pLSTMOutputExtractor(_print=False),
    pBLSTMLayer(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1, dropout=0.0),
    pLSTMOutputExtractor(_print=False),
    pBLSTMLayer(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1, dropout=0.0),
    pLSTMOutputExtractor(_print=False)
)

attention = FixedLengthSelfAttention(512, query_size=256, key_size=256, value_size=256)

p_output = layers(data)

print(f"Data input sequence length = {data.size(1)}\nAttention input sequence length = {p_output.size(1)}")
print(p_output.shape)

output = attention(p_output)
print(output.shape)
