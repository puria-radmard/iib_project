from typing import List
import torch, sys
from torch import nn
from numpy import ceil, floor


non_linearities = {
    'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU, 
    'softmax': lambda:nn.Softmax(dim=-1), 'none': nn.Identity
}


class MovingDNNLayer(nn.Module):

    """
        A DNN that slides over a sequence, and converts num_frames samples
        into a single vector, before shifting stride samples forward and repeating

        For example, consider num_frames = 3, stride = 2, mfcc_dims = 40, layer_dims = [256, 256]
        Then, the input sequence is of size [B, L, 40], i.e. one instance is 
            [A1, A2, A3, A4, A5, ..., AL], each of size 40

        The output is then
            [B1, B2, B3, B4, B5, ..., BL'], where:
                B1 = f(A1, A2, A3)
                B2 = f(A3, A4, A5)
                etc.
            where f(.) is a DNN of dimensions 120 -> 256 -> 256
            i.e. each Bi is of dimension 256

        Another MovingDNNLayer can then be fit onto this output sequence

        Nonlinearities are assumed same for each layer, and dropout is only applied at the end
    """

    def __init__(self, mfcc_dim, num_frames, stride, layer_dims, non_lin_func, dropout):

        super(MovingDNNLayer, self).__init__()

        layers = []
        layers.append(nn.Linear(mfcc_dim*num_frames, layer_dims[0]))
        layers.append(non_linearities[non_lin_func]())
        for i, hd in enumerate(layer_dims[:-1]):
            layers.append(nn.Linear(hd, layer_dims[i+1]))
            layers.append(non_linearities[non_lin_func]())
        layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

        self.bn = nn.BatchNorm1d(layer_dims[-1])
        self.stride = stride
        self.num_frames = num_frames

    def pad_sample(self, x):
        # For parallelised method to work, sequence length L needs to be:
        #     n * self.stride + self.num_frames
        B, L, D = x.shape
        remove_top = L - self.num_frames
        padding_length = self.stride - (remove_top % self.stride)

        if remove_top < 0:    # Not enough for one forward pass
            padding = torch.zeros(B, abs(remove_top), D)
        else:
            padding = torch.zeros(B, padding_length, D)

        return torch.cat([x, padding], dim=1)

    def make_slice(self, i):
        # Get all along batch dim but only self.num_frames samples along sequence dims
        return slice(None), slice(i*self.stride, i*self.stride + self.num_frames)

    def parallel_stack(self, x):
        # Stack all the slices that will be forward passed
        # Makes computation easier I think

        # Get the number of slices we need
        B, L, D = x.shape
        remove_top = L - self.num_frames
        num_slices = 1 + int(remove_top/self.stride) # should be a whole number given we've done pad_sample

        # Get the windows that will be passed
        slices = [x[self.make_slice(i)] for i in range(num_slices)]
        
        # Flatten them in the right way to get the input
        flattened_slices = [sl.view(B, -1) for sl in slices]

        # Now stack them to make it inputtable to self.layers
        return torch.stack(flattened_slices, dim = 1)
    
    def forward(self, x):
        padded_x = self.pad_sample(x)
        stacked_x = self.parallel_stack(padded_x)
        try:
            layers_output = self.layers(stacked_x)
        except:
            import pdb; pdb.set_trace()

        layers_output = layers_output.transpose(1, 2)
        layers_output = self.bn(layers_output)
        layers_output = layers_output.transpose(1, 2)

        return layers_output


class TDNNLayer(nn.Module):

    """
        One TDNN layer
            Takes input sequence [A1, A2, A3, ..., AL] of shape [N, L, mfcc_dim]
            Outputs sequence of shape [B1, B2, B3, ..., BL'] of shape [N, L', output_dim]

        Each Bi = f(A[stride * i + ci] for ci in context_idxs) s.t. i + ci is in 0,...,L-1
            i.e. does not go off bounds

        Dropout at end
    """
    
    def __init__(self, mfcc_dim, output_dim, context_idxs, non_lin_func, stride, dropout):

        super(TDNNLayer, self).__init__()
        
        # Save variable for methods
        self.stride = stride
        # Sorted list to maintain order
        self.context_idxs = sorted(list(context_idxs))

        # Layer takes in all context samples => input size is multiplied as such
        self.layers = nn.Sequential(
            nn.Linear(mfcc_dim*len(context_idxs), output_dim),
            non_linearities[non_lin_func](),
            nn.Dropout(dropout)
        )
        self.bn = nn.BatchNorm1d(output_dim)

    def get_viable_idxs(self, seq_len):
        """
            Get the viable indices, which will be a consecutive set of numbers
            the length of the output sequence is the number of viable indices
              e.g. if the input sequence is of size 15, self.context_idxs = [-1, +2], and self.stride = 1
                   we have viable_indices = [1, 2, ..., 12]
              e.g.2 if the input sequence is of size 15, self.context_idxs = [0, 1, 2, 3], and self.stride = 1
                    we have viable_indices = [0, 1, 2, ..., 11]
            This also illustrates why a stride of >1 might make things less well defined
        """
        l_bound = - min(self.context_idxs)
        u_bound = seq_len - max(self.context_idxs)
        viable_idx = list(range(l_bound, u_bound, self.stride))
        return viable_idx

    def index_context(self, x, idx):
        """
            For the given index, from the output of self.get_viable_idxs, index the relevant input samples,
            then flatten them
        """
        B, L, D = x.shape
        indices = [idx + ci for ci in self.context_idxs]
        # This should be of size B, 
        return x[:,indices,:].view(B, -1)

    def form_layer_input(self, x):
        """
            Form the input to our layers:
                1. Get the input size to the layers = the output size
                2. For each index in that range get the relevant context samples
                3. Stack them to relevant size
        """
        B, L, D = x.shape
        viable_indices = self.get_viable_idxs(L)
        return torch.stack([self.index_context(x, vi) for vi in viable_indices], dim = 1)

    def forward(self, x):
        layers_input = self.form_layer_input(x)
        layers_output = self.layers(layers_input)

        layers_output = layers_output.transpose(1, 2)
        layers_output = self.bn(layers_output)
        layers_output = layers_output.transpose(1, 2)

        return layers_output


class TDNNPadding(nn.Module):
    """
        We want to make sure the output of a succession of TDNNLayers will have at least one sample
        We use the layer contexts in order to get the padding on the input we require, then administer that padding

        See __init__ for cascade method, which only needs to be done once
    """

    def __init__(self, tdnn_layers: List[TDNNLayer]):
        min_size = 1
        for layer in tdnn_layers[::-1]:
            # How wide across are the input samples used to generate one output sample
            context_diameter = max(layer.context_idxs) - min(layer.context_idxs) + 1

            # How many samples from the input we need to generate the full output
            min_size = context_diameter + layer.stride * (min_size - 1)

        # Save for later - we pad sequences to get to at least this size
        self.min_size = min_size

        super(TDNNPadding, self).__init__()

    def forward(self, x):
        B, L, D = x.shape
        padding_required = self.min_size - L
        
        if padding_required <= 0:
            return x
        else:
            left_padding = torch.zeros([B, int(floor(padding_required/2)), D])
            right_padding = torch.zeros([B, int(ceil(padding_required/2)), D])
            return torch.cat([left_padding, x, right_padding], dim = 1)


if __name__ == '__main__':

    # Make an usually lengthed batch
    data = torch.randn(32, 17, 40)

    if sys.argv[1] == 'move':
        layers = [
            MovingDNNLayer(40, 10, 2, [256], 'relu', 0)
        ]
    elif sys.argv[1] == 'tdnn':
        tdnn_layers = [
            TDNNLayer(40, 256, [-2, -1, 0, 1, 2], 'relu', 3, 0),
            TDNNLayer(256, 256, [-4, 0, 4], 'relu', 1, 0),
            TDNNLayer(256, 256, [-7, 0, 7], 'relu', 1, 0),
        ]
        layers = [TDNNPadding(tdnn_layers)] + tdnn_layers

    class PrintLayer(nn.Module):
        def forward(self, x):
            print(x.shape)
            return x

    model = nn.Sequential()
    for i, layer in enumerate(layers):
        model.add_module(f'layer-{i}', layer)
        model.add_module(f'print-{i}', PrintLayer())

    print(model)
    output = model(data)
