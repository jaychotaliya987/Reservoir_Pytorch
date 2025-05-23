import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from typing import Optional, Callable, Type, Union

class SineSquare(Dataset):
    """
    This is a dataset of randomly arranged discretized Sine and Square wave. It outputs a torch.tensor type dataset. 
    args:
        param:sample_len - length of the dataset to be created`
    """

    def __init__(self, sample_len: int,
                 discretization: int = 8,
                 normalize: bool = False,
                 dtype: torch.dtype = torch.float32,
                 seed: int = None):

        self.discretization = 8
        self.sample_len = int(sample_len/self.discretization)
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
        return torch.sin(torch.arange(0, 2*torch.pi, 2*torch.pi/self.discretization))

    def _square(self):
        return torch.sign(self._sin())

    def _arrange(self):
        # Generate base waveforms
        sin_wave = self._sin()  # shape: [discretization]
        square_wave = self._square()  # shape: [discretization]

        # Create random selection mask
        map = torch.rand(self.sample_len) > 0.5  # shape: [sample_len]

        # Vectorized data construction
        self.data = torch.where(
            map.unsqueeze(1),  # shape: [sample_len, 1]
            sin_wave.unsqueeze(0).expand(self.sample_len, -1),
            square_wave.unsqueeze(0).expand(self.sample_len, -1)
        )  # shape: [sample_len, discretization]

        # Proper label generation (1:1 with data points)
        self.label = torch.where(
            map.unsqueeze(1).expand(-1, self.discretization),
            torch.zeros(self.sample_len, self.discretization, dtype=torch.int),
            torch.ones(self.sample_len, self.discretization, dtype=torch.int)
        ).flatten()  # shape: [sample_len * discretization]

        if self.normalize:
            self.data = self._normalize(self.data)

        return self.data, self.label


### ------------ Yankers ----------------- ###

    def get_all(self):
        """
        Returns whole dataset at once 
        """
        return self.data, self.label
