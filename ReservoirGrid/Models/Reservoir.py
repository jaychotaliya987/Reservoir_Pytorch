import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Reservoir(nn.Module):
    def __init__(self, input_dim, reservoir_dim, output_dim, 
                 spectral_radius=0.9, leak_rate=0.3, sparsity=0.9, 
                 input_scaling=1.0, noise_level=0.01):
        super(Reservoir, self).__init__()
        self.reservoir_dim = reservoir_dim
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.noise_level = noise_level
        
        # Initialize input weights with proper scaling
        self.W_in = torch.randn(reservoir_dim, input_dim) * input_scaling
        
        # Initialize sparse reservoir weights
        self.W = torch.rand(reservoir_dim, reservoir_dim) * 2 - 1
        # Create sparse mask
        mask = torch.rand(reservoir_dim, reservoir_dim) > sparsity
        self.W *= mask.float()
        
        # Scale spectral radius properly
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
        if reset_state:
            self.reservoir_state = torch.zeros_like(self.reservoir_state)
            self.reservoir_states = []
        
        device = u.device
        self.W_in = self.W_in.to(device)
        self.W = self.W.to(device)
        self.reservoir_state = self.reservoir_state.to(device)
        
        for t in range(u.size(0)):
            # Add small noise for regularization
            noise = torch.randn_like(self.reservoir_state) * self.noise_level
            
            # Update reservoir state with leaky integration
            input_term = torch.matmul(self.W_in, u[t])
            recurrent_term = torch.matmul(self.W, self.reservoir_state)
            new_state = self.activation(input_term + recurrent_term + noise)
            
            self.reservoir_state = (1 - self.leak_rate) * self.reservoir_state + \
                                  self.leak_rate * new_state
            self.reservoir_states.append(self.reservoir_state.clone())
        
        return self.readout(torch.stack(self.reservoir_states))

    def train_readout(self, inputs, targets, alpha=1e-6):
        """Train readout with ridge regression"""
        # First collect all reservoir states
        with torch.no_grad():
            self.forward(inputs, reset_state=True)
            X = torch.stack(self.reservoir_states)
            y = targets
        
        # Convert to numpy for ridge regression
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # More efficient ridge regression implementation
        X_T = X_np.T
        I = np.eye(self.reservoir_dim) * alpha
        solution = np.linalg.pinv(X_T @ X_np + I) @ X_T @ y_np
        
        # Update readout weights
        with torch.no_grad():
            self.readout.weight.data = torch.tensor(solution.T, dtype=torch.float32, device=inputs.device)
            self.readout.bias.data.zero_()

    def predict(self, initial_input, steps, teacher_forcing=None, warmup=0):
        """Predict future steps with optional teacher forcing and warmup"""
        predictions = []
        current_input = initial_input[-1]
        
        with torch.no_grad():
            # Warmup phase to stabilize reservoir state
            for _ in range(warmup):
                new_state = self.activation(
                    torch.matmul(self.W_in, current_input) + 
                    torch.matmul(self.W, self.reservoir_state)
                )
                self.reservoir_state = (1 - self.leak_rate) * self.reservoir_state + \
                                     self.leak_rate * new_state
            
            # Prediction phase
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