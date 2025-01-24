import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

class ESN(nn.Module):
    def __init__(self, input_dim, reservoir_dim, output_dim, spectral_radius=0.95):
        """
        input_dim: Dimension of input sequence
        reservoir_dim: Dimension of reservoir
        output_dim: Dimension of output sequence, 
                    For the regression tasks, output_dim = 1
        spectral_radius: Spectral radius of reservoir matrix
        """

        super(ESN, self).__init__()
        # Initialize fixed reservoir and input weights
        self.W_in = torch.randn(reservoir_dim, input_dim) * 0.1  # Scale input weights
        self.W = torch.randn(reservoir_dim, reservoir_dim)       # Reservoir weights
        # Scale reservoir weights to ensure echo state property
        max_eigenvalue = max(abs(torch.linalg.eigvals(self.W).real))
        self.W *= spectral_radius / max_eigenvalue
        
        # Output layer (trainable)
        self.readout = nn.Linear(reservoir_dim, output_dim, bias=True)
        
        # Nonlinearity for reservoir
        self.activation = torch.tanh
        
        # Reservoir state
        self.reservoir_state = torch.zeros(reservoir_dim)
    
    def forward(self, u):
        """
        Forward pass through the reservoir and readout layer.
        :param u: Input sequence (T x input_dim)
        :return: Output (T x output_dim)
        """
        outputs = []
        for t in range(u.size(0)):  # Iterate through time steps
            # Update reservoir state
            self.reservoir_state = self.activation(
                torch.matmul(self.W_in, u[t]) + torch.matmul(self.W, self.reservoir_state)
            )
            # Compute output
            y = self.readout(self.reservoir_state)
            outputs.append(y)
        return torch.stack(outputs)
    
    def freeze_reservoir(self):
        # Ensure W_in and W are fixed
        self.W_in.requires_grad = False
        self.W.requires_grad = False
