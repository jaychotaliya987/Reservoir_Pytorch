import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset


class ESN(nn.Module):
    def __init__(self, input_dim, reservoir_dim, output_dim, spectral_radius=0.95):
        """
        input_dim: Dimension of input sequence
        reservoir_dim: Dimension of reservoir
        output_dim: Dimension of output sequence, 
                    For regression tasks, output_dim = 1
        spectral_radius: Spectral radius of reservoir matrix
        """
        super(ESN, self).__init__()
        
        # Initialize fixed reservoir and input weights
        self.W_in = torch.randn(reservoir_dim, input_dim) * 0.1  # Scale input weights
        self.W = torch.randn(reservoir_dim, reservoir_dim)       # Reservoir weights
        
        # Scale reservoir weights to ensure echo state property
        with torch.no_grad():
            max_eigenvalue = max(abs(torch.linalg.eigvals(self.W).real))
            self.W *= spectral_radius / max_eigenvalue
            
        # Output layer (trainable)
        self.readout = nn.Linear(reservoir_dim, output_dim, bias=True)
        
        # Nonlinearity for reservoir
        self.activation = torch.tanh
        
        # Reservoir state (initialized as a parameter to ensure it moves with the model's device)
        self.register_buffer('reservoir_state', torch.zeros(reservoir_dim))

    def forward(self, u):
        """
        Forward pass through the reservoir and readout layer.
        :param u: Input sequence (T x input_dim)
        :return: Output (T x output_dim)
        """
        outputs = []  # Use a list to accumulate outputs
        device = u.device
        self.reservoir_states = torch.tensor([]).to(device)  # Ensure reservoir state is on the same device
        self.reservoir_state = self.reservoir_state.to(device)  # Ensure reservoir state is on the same device

        for t in range(u.size(0)):  # Iterate through time steps
            # Update reservoir state
            self.reservoir_state = self.activation(
                torch.matmul(self.W_in.to(device), u[t]) +
                torch.matmul(self.W.to(device), self.reservoir_state))
            
            self.reservoir_states = torch.cat((self.reservoir_states, self.reservoir_state.unsqueeze(0)), dim=0)
            
            # Compute output
            y = self.readout(self.reservoir_state)
            outputs.append(y)
        
        return torch.stack(outputs)  # Convert list to tensor
    

    def Train(self, dataset: torch.tensor, targets : torch.tensor, epochs: int, lr: float, 
              criterion=nn.MSELoss, optimizer=optim.Adam, print_every=10):
        """
        Trains the model
        :param dataset: Dataset for training
        :param epochs: Number of epochs
        :param lr: Learning rate
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
                print(f'Epoch {epochs}, Iteration {epoch}, Loss: {loss.item()}')
        return losses
        
    
                                                        
    def Predict(self, input, steps):
        """
        Predict future outputs using an ESN in autonomous mode.
        Args:
            input (torch.Tensor): Initial input tensor
            steps (int): Number of time steps to predict
        Returns:
            torch.Tensor: Predicted outputs of shape (steps, 1)
        """
        device = input.device
        preds = input[-1]
        predictions = []

        for _ in range(steps):
            r_state_last = self.res_state()
            r_state_last = torch.tanh(torch.matmul(self.res_w().to(device), r_state_last.to(device))
                                       + torch.matmul(self.w_in().to(device), preds.to(device)))
            
            pred = self.readout(r_state_last)
            self.update_reservoir(r_state_last)
            predictions.append(pred)

        return torch.cat(predictions, dim=0)
                    
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

####___________Plots_______________####
  
    def Plots(self, u, future = int, memory = int):
        """
        Plots the model's predictions
        :param u: Input sequence (T x input_dim)
        :param future: Number of future predictions
        :param memory: Number of previous time steps to remember
        """
        predictions = self.Predictions(u, future, memory)
        plt.plot(u, label='Input')
        plt.plot(range(len(u), len(u) + future), predictions, label='Predictions')
        plt.legend()
        plt.show()
        
        return predictions


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