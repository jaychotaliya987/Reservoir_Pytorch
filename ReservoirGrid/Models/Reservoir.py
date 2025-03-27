import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Reservoir(nn.Module):
    def __init__(self, input_dim, reservoir_dim, output_dim, spectral_radius=0.1, leak_rate=0.3):
        super(Reservoir, self).__init__()
        self.reservoir_dim = reservoir_dim
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius

        # Initialize input weights
        self.W_in = torch.randn(reservoir_dim, input_dim) * 0.1
        
        # Initialize sparse reservoir weights
        self.W = torch.rand(reservoir_dim, reservoir_dim) * 2 - 1
        sparsity = 0.95
        self.W[torch.rand(reservoir_dim, reservoir_dim) < sparsity] = 0
        
        # Scale spectral radius
        eigenvalues, _ = torch.linalg.eig(self.W)  # Get full eigenvalues
        max_eigenvalue = torch.max(torch.abs(eigenvalues))  # Get largest magnitude
        self.W *= spectral_radius / max_eigenvalue  # Proper scaling


        print("Max Eigenvalue After Scaling:", max_eigenvalue)


        # Readout layer
        self.readout = nn.Linear(reservoir_dim, output_dim)
        self.activation = torch.tanh
        
        # Initialize state
        self.register_buffer('reservoir_state', torch.zeros(reservoir_dim))
        
        # This will store states during forward pass
        self.reservoir_states = []

    def forward(self, u, reset_state=True):
        if reset_state:
            self.reservoir_state = torch.zeros_like(self.reservoir_state)
            self.reservoir_states = []  # Reset states storage
        
        device = u.device
        self.W_in = self.W_in.to(device)
        self.W = self.W.to(device)
        self.reservoir_state = self.reservoir_state.to(device)
        
        for t in range(u.size(0)):
            # Update reservoir state with leaky integration
            new_state = self.activation(
                torch.matmul(self.W_in, u[t]) + 
                torch.matmul(self.W, self.reservoir_state)
            )
            self.reservoir_state = (1 - self.leak_rate) * self.reservoir_state + \
                                   self.leak_rate * new_state
            self.reservoir_states.append(self.reservoir_state.clone())
        
        return self.readout(torch.stack(self.reservoir_states))

    def train_readout(self, inputs, targets, alpha=1e-6):
        """Train readout with ridge regression"""
        # First collect all reservoir states
        with torch.no_grad():
            self.forward(inputs, reset_state=True)  # This populates self.reservoir_states
            X = torch.stack(self.reservoir_states)
            y = targets
        
        # Ridge regression solution
        X, y = X.cpu().numpy(), y.cpu().numpy()
        I = np.eye(X.shape[1]) * alpha  # Ensure correct regularization
        solution = np.linalg.solve(X.T @ X + I, X.T @ y)

        
        # Update readout weights
        self.readout.weight.data = torch.tensor(solution.T, dtype=torch.float32, device=inputs.device)
        self.readout.bias.data = torch.zeros_like(self.readout.bias.data)


    def predict(self, initial_input, steps, teacher_forcing=None):
        """Predict future steps with optional teacher forcing"""
        predictions = []
        current_input = initial_input[-1]
        
        with torch.no_grad():
            for step in range(steps):
                new_state = self.activation(
                    torch.matmul(self.W_in, current_input) + 
                    torch.matmul(self.W, self.reservoir_state)
                )
                self.reservoir_state = (1 - self.leak_rate) * self.reservoir_state + \
                                       self.leak_rate * new_state
                pred = self.readout(self.reservoir_state)
                predictions.append(pred)
                
                # Teacher forcing (optional)
                if teacher_forcing is not None and step < len(teacher_forcing):
                    current_input = teacher_forcing[step]
                else:
                    current_input = pred  # Autonomous feedback
        
        return torch.stack(predictions)
