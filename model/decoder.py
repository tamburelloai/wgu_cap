from torch import nn


class Decoder(nn.Module):
    def __init__(self, d_model, n_embed):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(
            in_features=d_model,
            out_features=n_embed
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        return self.decoder(x)


