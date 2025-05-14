import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from typing import Optional, Callable, Type, Union

class SineSquare(Dataset):
    """
    This is a dataset of randomly arranged discretized Sine and Square wave. It outputs a torch.tensor type dataset. 
    """

    def __init__(self, sample_len: int,
                 discretization: int = 8,
                 normalize: bool = False,
                 dtype: torch.dtype = torch.float32,
                 seed: int = None):
        self.sample_len = sample_len
        self.discretization = 8
        self.normalize = normalize
        self.seed = seed
        self.dtype = dtype

        if seed is not None:
            torch.manual_seed(seed)

        self.data, self.label = self._arrange()

### --------- overrides ------------ ###


    def __len__(self):
        """
        Length
        :return:
        """
        return self.sample_len
 
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        return self.data[idx], self.label[idx]

### --------- Generation ----------- ###
    def _sin(self):
        return torch.arange(0, 2*torch.pi, torch.pi/self.discretization)

    def _square(self):
        return torch.sign(self._sin())

    def _arrange(self):
        """
        Arranges sin and square waves randomly. Optionally normalizes between [0,1]
        """

        map = torch.rand(self.sample_len) > 0.5
        data_list = []
        label_list = []

        for _ in range(self.sample_len):
            if map[_]:
                data_list.append(self._sin())
                label_list.append(0)
            else:
                data_list.append(self._square())
                label_list.append(1)

        self.data = torch.cat(data_list, dim=0)
        self.label = torch.tensor(label_list)
        # Normalizing the data If necessory
        if self.normalize:
            data = self._normalize(self.data)

        return self.data, self.label

    def _normalize(self, Data):
        # Normalizing the data in range [0,1]
        return (Data - Data.min()) / (Data.max() - Data.min())

### ------------ Yankers ----------------- ###

    def get_all(self):
        """
        Returns whole dataset at once
        """
        return self.data, self.label
