import torch
from torch import nn
from torch.nn import functional as F


non_linearities = {
    'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU, 
    'softmax': lambda:nn.Softmax(dim=-1), 'none': nn.Identity
}


class SelfAttention(nn.Module):

    """
        (seq_. are all the same, but used to differentiate)

        k [batch, seq_k, key_size]
        q [batch, seq_q, query_size]
        v [batch, seq_v, value_size]

        e [batch, seq_q, query_size] @ ([query_size, key_size] @ [batch, seq_k, key_size]).permute(0, 2, 1)
            = [batch, seq_q, query_size] @ [batch, query_size, seq_k]
            = [batch, seq_q, seq_k]

        alpha = softmax e over dim -1 (seq_k, as required in definition)
            = [batch, seq_q, seq_k]

        unweighted = [value_size, value_size] @ [batch, seq_v, value_size]
            = [batch, seq_v, value_size]
        
        u_unsummed = [batch, seq_q, seq_k] @ [batch, seq_v, value_size]
            = [batch, seq_q, ]
    """

    def __init__(self, input_size, num_heads, query_size, key_size, value_size):
        
        super(SelfAttention, self).__init__()

        self.Wk = nn.Linear(input_size, num_heads*key_size, bias=False)
        self.Wq = nn.Linear(input_size, num_heads*query_size, bias=False)
        self.Wv = nn.Linear(input_size, num_heads*value_size, bias=False)
        self.Wo = nn.Linear(value_size*num_heads, value_size*num_heads, bias=False)
        
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.num_heads = num_heads

        self.temperature = self.key_size ** 0.5
        self.softmax = lambda x: F.softmax(x / self.temperature, dim=-1)

    def push_head_dim(self, tensor):
        head_pushed = tensor.view(tensor.size(0), tensor.size(1), self.num_heads, -1)
        return head_pushed.permute(0, 2, 1, 3)

    def get_tensors(self, x):
        k = self.push_head_dim(self.Wk(x))
        q = self.push_head_dim(self.Wq(x))
        v = self.push_head_dim(self.Wv(x))
        return k, q, v

    def get_alignments(self, k, q):
        e = q @ k.permute(0, 1, 3, 2)
        return self.softmax(e)

    def get_output_sequence(self, alignments, v):
        weighted = alignments @ v
        return weighted.contiguous().view(weighted.size(0), weighted.size(2), self.num_heads*self.value_size)
        
    def forward(self, x):
        k, q, v = self.get_tensors(x)
        alignments = self.get_alignments(k, q)
        output_sequence = self.get_output_sequence(alignments, v)
        return self.Wo(output_sequence)


class SelfAttentionTranformerLayer(nn.Module):

    """
        Same as the AIAYN paper, except we depend on the listener layer to give us the positional
        encoding that are handcrafted in AIAYN

        d_model makes things simple & cascadeable - used by paper as well
            i.e. Ensure that sublayers are all d_model, allowing residual connections

        Default FFN activation: relu then no activation
        Default d_model = 512
        Default hidden_sizes = [2048] only
    """

    def __init__(self, d_model, num_heads, hidden_sizes, non_linearity='relu'):

        super(SelfAttentionTranformerLayer, self).__init__()

        # Input size and D/K/Q sizes defined by d_model and number of self attention heads
        self.input_size = d_model
        self.num_heads = num_heads
        self.d_model = d_model
        self.d = d_model // num_heads
        self.non_linearity = non_linearity
        self.hidden_sizes = hidden_sizes

        # As above
        self.self_attention = SelfAttention(self.input_size, self.num_heads, self.d, self.d, self.d)
        self._make_feedforward_network()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def _make_feedforward_network(self):
        # Define the feedforward - as before, these need to be 'residualable' => d_model throughout        
        self.feed_forward = nn.Sequential()
        non_lin_method = non_linearities[self.non_linearity]

        self.feed_forward.add_module('fc_0', nn.Linear(self.d_model, self.hidden_sizes[0]))
        self.feed_forward.add_module('af_0', non_lin_method())
        
        for i, hs in enumerate(self.hidden_sizes[1:], 1):
            self.feed_forward.add_module(f'fc_{i}', nn.Linear(self.hidden_sizes[i-1], hs))
            self.feed_forward.add_module(f'af_{i}', non_lin_method())

        self.feed_forward.add_module('fc_out', nn.Linear(self.hidden_sizes[-1], self.d_model))
        # No activation at the end

    def forward(self, x):
        self_attention_output = self.self_attention(x)
        add_and_norm_output_1 = self.layer_norm_1(self_attention_output + x)
        feed_forward_network_output = self.feed_forward(add_and_norm_output_1)
        add_and_norm_output_2 = self.layer_norm_2(feed_forward_network_output + add_and_norm_output_1)
        return add_and_norm_output_2
