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
