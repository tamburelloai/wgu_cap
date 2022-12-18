import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset





class TorchDS(Dataset):
    def __init__(self, data, seqlen=12):
        self.data = data
        self.seqlen = seqlen

    def __len__(self):
        return len(self.X) - (self.seqlen)

    def __getitem__(self, index):
        return (self.X[index:index+self.seqlen], self.X[index+self.seqlen, -1])


