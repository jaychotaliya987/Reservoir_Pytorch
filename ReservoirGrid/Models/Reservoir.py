import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class ESN(nn.Module):
    def __init__(self, input_dim, reservoir_dim, output_dim, spectral_radius=0.9, leak_rate=0.3):
        super(ESN, self).__init__()
        self.reservoir_dim = reservoir_dim
        self.leak_rate = leak_rate

        # Initialize input weights
        self.W_in = torch.randn(reservoir_dim, input_dim) * 0.1
        
        # Initialize sparse reservoir weights
        self.W = torch.rand(reservoir_dim, reservoir_dim) * 2 - 1
        sparsity = 0.9  # 90% sparse
        self.W[torch.rand(reservoir_dim, reservoir_dim) < sparsity] = 0
        
        # Scale spectral radius
        max_eigenvalue = max(abs(torch.linalg.eigvals(self.W).real))
        self.W *= spectral_radius / max_eigenvalue

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
        I = np.eye(X.shape[1])
        solution = np.linalg.solve(X.T @ X + alpha * I, X.T @ y)
        
        # Update readout weights
        self.readout.weight.data = torch.tensor(solution.T, dtype=torch.float32, device=inputs.device)
        self.readout.bias.data.zero_()

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

# Generate synthetic data
time = torch.linspace(0, 10, 1000)
data = torch.sin(time).unsqueeze(1)  # (1000, 1)

# Normalize data to [-1, 1]
data = (data - data.min()) / (data.max() - data.min()) * 2 - 1

# Split into train and test
train_data, test_data = data[:800], data[800:]

# Initialize ESN
esn = ESN(input_dim=1, reservoir_dim=100, output_dim=1, 
          spectral_radius=0.9, leak_rate=0.3)

# Train the readout
esn.train_readout(train_data, train_data)

# Warm up reservoir with last 50 points of training data
warmup_length = 50
with torch.no_grad():
    _ = esn(train_data[-warmup_length:], reset_state=True)
    
    # Predict next 100 steps
    predictions = esn.predict(train_data[-1:], steps=100)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(torch.cat([train_data[-50:], test_data[:100]]).numpy(), label='True')
plt.plot(range(50, 150), predictions.numpy(), '--', label='Predicted')
plt.axvline(x=50, color='r', linestyle=':', label='Prediction Start')
plt.legend()
plt.show()
