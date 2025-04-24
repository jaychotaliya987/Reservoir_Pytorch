'''
Main Reservoir class for the ReservoirGrid project.
This class implements a reservoir computing model with the following features:
- Reservoir state update with leaky integration
- Readout layer for prediction
- Training of the readout layer with ridge regression
- Optional training of the reservoir with backpropagation
- Prediction with optional teacher forcing
- Saving and loading of the model
- Echo state property checks
- Reservoir state and weights visualization and control
- Device management (CPU/GPU)
'''

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Optional, Callable, Type, Union

# Default device (can be overridden)
_DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DEFAULT_DTYPE = torch.float32


class Reservoir(nn.Module):
    def __init__(self, input_dim, 
                 reservoir_dim, 
                 output_dim, 
                 spectral_radius=0.9, 
                 leak_rate=0.3,
                 sparsity=0.9, 
                 input_scaling=1.0, 
                 noise_level=0.01):
        

        """
        Initialize the Reservoir class.
        Args:
            :param input_dim: Dimension of the input
            :param reservoir_dim: Dimension of the reservoir
            :param output_dim: Dimension of the output
            :param spectral_radius: Spectral radius of the reservoir weights
            :param leak_rate: Leak rate for the reservoir state update
            :param sparsity: Sparsity of the reservoir weights
            :param input_scaling: Scaling factor for the input weights
            :param noise_level: Noise level for the reservoir state update
        """

        
        super(Reservoir, self).__init__()
        self.reservoir_dim = reservoir_dim
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.noise_level = noise_level
        self.sparsity = sparsity
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.input_scaling = input_scaling

        # Initialize input weights with proper scaling
        self.W_in = torch.randn(reservoir_dim, input_dim) * input_scaling

        # Initialize sparse reservoir weights
        self.W = torch.rand(reservoir_dim, reservoir_dim) * 2 - 1
        # Create sparse mask
        mask = torch.rand(reservoir_dim, reservoir_dim) > sparsity
        self.W *= mask.float()

        # Scale spectral radius 
        eigenvalues = torch.linalg.eigvals(self.W)
        current_spectral_radius = torch.max(torch.abs(eigenvalues))
        self.W *= (spectral_radius / current_spectral_radius)

        # Readout layer
        self.readout = nn.Linear(reservoir_dim, output_dim)
        self.activation = torch.tanh

        # Initialize state
        self.register_buffer('reservoir_state', torch.zeros(reservoir_dim))
        self.reservoir_states = []

    def forward(self, u, reset_state=True):
        """
        Forward pass through the reservoir and readout layer.
        :param u: Input sequence (T x input_dim)
        :return: Output (T x output_dim)
        """
        if reset_state:
            self.reservoir_state = torch.zeros_like(self.reservoir_state)
            self.reservoir_states = []
        
        device = u.device
        self.W_in = self.W_in.to(device)
        self.W = self.W.to(device)
        self.reservoir_state = self.reservoir_state.to(device)

        for t in range(u.size(0)):
            #Small noise for regularization
            noise = torch.randn_like(self.reservoir_state) * self.noise_level

            # Update reservoir state with leaky integration
            input_term = torch.matmul(self.W_in, u[t])
            recurrent_term = torch.matmul(self.W, self.reservoir_state)
            new_state = self.activation(input_term + recurrent_term + noise)

            self.reservoir_state = (1 - self.leak_rate) * self.reservoir_state + \
                                  self.leak_rate * new_state
            self.reservoir_states.append(self.reservoir_state.clone())

        return self.readout(torch.stack(self.reservoir_states))

    def train_readout(self, inputs, targets, warmup = None ,alpha=1e-6):
        """Train readout with ridge regression"""
        
        if warmup is not None:
            inputs = inputs[warmup:]
            targets = targets[warmup:]
        

        with torch.no_grad():
            self.forward(inputs, reset_state=True)
            X = torch.stack(self.reservoir_states)
            y = targets

        # Convert to numpy for ridge regression
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()

        X_T = X_np.T
        I = np.eye(self.reservoir_dim) * alpha
        solution = np.linalg.pinv(X_T @ X_np + I) @ X_T @ y_np

        # Update readout weights
        with torch.no_grad():
            self.readout.weight.data = torch.tensor(solution.T, dtype=torch.float32, device=inputs.device)
            self.readout.bias.data.zero_()

    def Train(self, dataset: torch.tensor, targets : torch.tensor, epochs: int, lr: float, 
              criterion=nn.MSELoss, optimizer=optim.Adam, print_every=10):
        """
        Trains the model with BP, for additional accuarcy.
        :param dataset: Dataset for training
        :param epochs: Number of epochs
        :param lr: Learning rate
        :param criterion: Loss function (default: MSELoss)
        :param optimizer: Optimizer (default: Adam)
        :param print_every: Print loss every n epochs
        :return: losses Tensor for plotting
        """
        # Define loss function and optimizer
        criterion = criterion()
        optimizer = optimizer(self.parameters(), lr=lr)
        
        # Move model to the same device as the dataset
        device = dataset.device
        self.to(device)

        # losses Tensor for plotting
        losses = torch.tensor([]).to(device)

        for epoch in range(epochs):   
            optimizer.zero_grad()  
            output = self(dataset)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            losses = torch.cat((losses, loss.unsqueeze(0)), dim=0)
            if epoch % print_every == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
            if epoch == epochs - 1:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
        return losses
        
    def predict(self, input, steps, teacher_forcing=None, warmup=0):
        """Predict future steps with optional teacher forcing and warmup"""
        predictions = []
        current_input = input[-1]

        with torch.no_grad():
            # Warmup phase to stabilize reservoir state
            for _ in range(warmup):
                new_state = self.activation(
                    torch.matmul(self.W_in, current_input) + 
                    torch.matmul(self.W, self.reservoir_state)
                )
                self.reservoir_state = (1 - self.leak_rate) * self.reservoir_state + \
                                     self.leak_rate * new_state

            # Autonomous Predictions
            for step in range(steps):
                new_state = self.activation(
                    torch.matmul(self.W_in, current_input) + 
                    torch.matmul(self.W, self.reservoir_state)
                )
                self.reservoir_state = (1 - self.leak_rate) * self.reservoir_state + \
                                     self.leak_rate * new_state
                pred = self.readout(self.reservoir_state)
                predictions.append(pred)

                # Teacher forcing logic
                if teacher_forcing is not None and step < len(teacher_forcing):
                    current_input = teacher_forcing[step]
                else:
                    current_input = pred

        return torch.stack(predictions)

    def update_reservoir(self, u):
        """
        Update the reservoir state using the input sequence.
        :param u: Input sequence (T x input_dim)
        """
        device = u.device
        self.reservoir_state = u
        self.reservoir_states = torch.cat((self.reservoir_states, self.reservoir_state.unsqueeze(0)), dim=0)

####___________Echo State Property___________####

    def freeze_reservoir(self):
       # Ensure W_in and W are fixed
       self.W_in.requires_grad = False
       self.W.requires_grad = False

    def Unfreeze_reservoir(self):
        # Ensure W_in and W are trainable
        self.W_in.requires_grad = True
        self.W.requires_grad = True
    
    def Reset(self):
        """
        Resets the reservoir state
        """
        self.reservoir_state = torch.zeros_like(self.reservoir_state)
        self.reservoir_states = []

####___________Saving and Loading___________####

    def Save_model(self, path = str):
        """
        Saves the model
        :param path: Path to save the model
        """
        torch.save(self.state_dict(), path)
    
    def Load_model(self, path = str):
        """
        Loads the model
        :param path: Path to load the model
        """
        self.load_state_dict(torch.load(path))


####___________Get Methods___________####

    def res_states(self):
        return self.reservoir_states

    def res_state(self):
        return self.reservoir_state

    def readout_layer(self):
        return self.readout

    def res_w(self):
        return self.W
    
    def w_in(self):
        return self.W_in
    

