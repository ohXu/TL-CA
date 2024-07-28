from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
import time
import pandas as pd
import numpy as np
import torch


class CustomDataset(Dataset):
    def __init__(self, data):
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx][0]
        label = self.dataset[idx][1]
        data = torch.tensor(data.astype(np.float32))
        data = data.permute(2, 0, 1)
        label = torch.tensor(np.array(label, dtype=np.float32))
        return data, label
