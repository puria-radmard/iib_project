import torch
from torch import nn
from torch.nn import functional as F

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
        
        self.query_size = query_size
        self.key_size = key_size // num_heads
        self.value_size = value_size // num_heads
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
        summed = weighted.sum(dim=1)
        return summed
        
    def forward(self, x):
        k, q, v = self.get_tensors(x)
        alignments = self.get_alignments(k, q)
        output_sequence = self.get_output_sequence(alignments, v)
        return output_sequence
