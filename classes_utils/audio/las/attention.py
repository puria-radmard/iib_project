import torch
from torch import nn

class SelfAttentionBase(nn.Module):

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

    def __init__(self, input_size, query_size, key_size, value_size):
        super(SelfAttentionBase, self).__init__()
        self.Wk = nn.Linear(input_size, key_size, bias=False)
        self.Wq = nn.Linear(input_size, query_size, bias=False)
        self.Wv = nn.Linear(input_size, value_size, bias=False)
        self.Wa = nn.Linear(key_size, query_size, bias=False)
        self.Wb = nn.Linear(value_size, value_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def get_tensors(self, x):
        k = self.Wk(x)
        q = self.Wq(x)
        v = self.Wv(x)
        return k, q, v

    def get_alignments(self, k, q):
        e = q @ self.Wa(k).permute(0, 2, 1)
        return self.softmax(e)

    def get_output_sequence(self, alignments, v):
        # alignments = 0.1*(10*alignments).round()
        # alignments[:,0,0] = 0
        # import pdb; pdb.set_trace()
        unweighted = self.Wb(v)
        weighted = unweighted.unsqueeze(2) * alignments.permute(0, 2, 1).unsqueeze(-1)
        summed = weighted.sum(dim=1)
        return summed


class SequenceSelfAttention(SelfAttentionBase):
        
    def forward(self, x):
        k, q, v = self.get_tensors(x)
        alignments = self.get_alignments(k, q)
        output_sequence = self.get_output_sequence(alignments, v)
        return output_sequence


class FixedLengthSelfAttention(SelfAttentionBase):
        
    def forward(self, x):
        k, q, v = self.get_tensors(x)
        alignments = self.get_alignments(k, q)
        output_sequence = self.get_output_sequence(alignments, v)
        return output_sequence.sum(1)
