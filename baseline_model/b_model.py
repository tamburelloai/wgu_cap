import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class LinearRegression(nn.Module):
    def __init__(self, past, future):
        super(LinearRegression, self).__init__()
        self.network = nn.Linear(past, 1)
        self.future = future


    def forward(self,x:Tensor):
        yhat = []
        for i in range(self.future):
            yhat_i = self.network(x)
            yhat.append(yhat_i.view(-1))
            x = torch.cat([x.view(-1)[1:], yhat_i.view(-1)], dim=-1)
        return torch.stack(yhat, dim=-1)


