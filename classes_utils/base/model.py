from torch import nn

class ModelWrapper(nn.Module):

    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.midloop = False

    def forward(self, x, *args, **kwargs):
        encodings, decodings = self.model(x)
        if self.midloop:
            # Training, so return logits
            return encodings, decodings
        else:
            return {'last_logits': decodings, 'embeddings': encodings}


class MultitaskHead(nn.Module):
    """
        For multitask distillation/acquisition prediction
        
        Sizes in:
            prev model layers output: [N, C + 1]
        Sizes out:
            distillation output: [N, C]
            acquisition output: [N, 1]
        
        sigmoid_target: if True, acquisition output is passed through a Sigmoid

        see also: generate_speech_multitask_muzzle
    """
    
    def __init__(self, sigmoid_target, num_classes):
        super(MultitaskHead, self).__init__()

        # If sigmoid_target, acquisition output is passed through a Sigmoid
        self.acquisition_head = nn.Sigmoid() if sigmoid_target else nn.Identity()

        # C in the above description
        self.num_classes = num_classes

    def forward(self, x):
        
        # Double check this is the correct input to head
        num_preds = x.size(1)
        assert num_preds == self.num_classes + 1

        # Split the task predicitions as reqiured above
        distillation_output = x[:,:self.num_classes]
        acquisition_output = x[:,-1:]

        # Pass acquisition prediction through a sigmoid if needed
        acquisition_output = self.acquisition_head(acquisition_output)

        return distillation_output, acquisition_output