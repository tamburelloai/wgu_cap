import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from model.positional_encoding import PositionalEncoding
from model.encoder import Encoder
from model.decoder import Decoder
from torch.nn import functional as F

#TODO: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

class Transformer(nn.Module):
    def __init__(self, inpt_features, d_model, nhead,
                 d_hid, nlayers, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model,
                nhead,
                d_hid,
                dropout),
            nlayers
        )
        self.encoder = nn.Linear(1, d_model)
        self.decoder = nn.Linear(d_model, 1)

        self.projection = nn.Linear(inpt_features, 1)


    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = src.unsqueeze(-1)
        src = F.relu(self.encoder(src)) * math.sqrt(self.d_model)
        src = src.unsqueeze(-2)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output.squeeze(-1)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

