import torch 
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Optional, Callable, Type, Union
import qutip as qt

# Default device (can be overridden)
_DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DEFAULT_DTYPE = torch.float32


class CQOscRes(nn.Module):
    def __init__(self,
            eps_0: float,
            input_dim: int,
            h_tranculate:int, 
            omega:tuple,
            kappa:tuple,
            coupling: float,
            output_dim: int,
            device: Optional[Union[str, torch.device]] = None,
            dtype: torch.dtype = _DEFAULT_DTYPE):
        super(CQOscRes, self).__init__()

        self.eps_0 = eps_0
        self.input_dim = input_dim
        self.h_tranculate = h_tranculate
        self.omega = omega
        self.kappa = kappa
        self.coupling = coupling

        self.dtype = dtype
        self.device = torch.device(device) if device else _DEFAULT_DEVICE

        #Annihilator operators for each oscillators
        self.a = qt.tensor(qt.destroy(self.h_tranculate), qt.qeye(self.h_tranculate))
        self.b = qt.tensor(qt.qeye(self.h_tranculate), qt.destroy(self.h_tranculate))

        # Static Hamiltonian
        self.H_sttic = (self.omega[0] * self.a.dag() * self.a + 
                         self.omega[1] * self.b.dag() * self.b +
                         self.coupling * (self.a * self.b.dag() + self.a.dag() * self.b))


        self.time = np.linspace (0, 100e-9, 100)


    def H_drive(self, t):
        pass


    def forward(self, args):

        self.eps_a = 32
        self.eps_b = 23

        density_matrix = qt.mesolve()
        pass


    def train_readout():
        '''
        Performs Ridge regression in closed form.
        '''
        self.collected_states = []
        ridge_solver(collected_states, expected_states)

        pass
