from torch import nn


class Encoder(nn.Module):
    def __init__(self, n_embed, d_model):
        super(Encoder, self).__init__()
        self.encoder = nn.Embedding(
            num_embeddings=n_embed,
            embedding_dim=d_model
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        return self.encoder(x)
