import torch 
import matplotlib.pyplot as plt
import numpy
import plotly
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Optional, Callable, Type, Union

# Default device (can be overridden)
_DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DEFAULT_DTYPE = torch.float32


class CQOscRes(nn.Module):
    def __init__(self,
            eps_0: float,
            input_dim: int,
            max_occupation:int, 
            reservoir_dim: int,
            output_dim: int,
            spectral_radius: float = 0.9,
            leak_rate: float = 0.3,
            sparsity: float = 0.9,
            input_scaling: float = 1.0,
            noise_level: float = 0.01,
            activation: Callable = torch.tanh,
            device: Optional[Union[str, torch.device]] = None,
            dtype: torch.dtype = _DEFAULT_DTYPE):
        super(CQOscRes, self).__init__()
        
        self.eps_0 = eps_0
        self.input_dim = input_dim
        self.max_occupation = max_occupation

        self.dtype = dtype
        self.device = torch.device(device) if device else _DEFAULT_DEVICE

        eps_a = (torch.tensor(self.eps_0).expand(self.input_dim)
        rawDrive = 
        

    

    def forward():
        pass
    

    def train_readout():
        pass
