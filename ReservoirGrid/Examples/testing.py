import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import sys
import os

# Ensure the correct path to MackeyGlass module
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from Datasets.MackeyGlassDataset import MackeyGlassDataset
from Models.Echostate import ESN

print("Imports Done!\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## INITIALIZATION
# Generate Mackey-Glass dataset
Mglass1 = MackeyGlassDataset(1000, 5, tau=17, seed=0)

inputs, targets = Mglass1[0]
inputs = inputs.to(device) 
targets = targets.to(device)

esn = ESN(input_dim=1, reservoir_dim=500, output_dim=1, spectral_radius=0.95)
esn = esn.to(device)  

# Debug device placement
print(f"Model is on: {next(esn.parameters()).device}")
print(f"Inputs are on: {inputs.device}")
print(f"Targets are on: {targets.device}")

# Train the model
esn.freeze_reservoir()
losses = esn.Train(dataset=inputs, targets=targets, epochs=10, 
                   lr=0.001, criterion=nn.MSELoss, print_every=1)

# Plot training losses
plt.plot(losses.cpu().detach().numpy())
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()

# Predict future values
steps = 200
predictions = esn.Predict(inputs, steps).cpu().detach().numpy()

inputs_plot = inputs[:-200]  # Keep last 200 for prediction comparison

plt.figure(figsize=(10, 5))

# Training data
plt.plot(range(len(inputs_plot)), inputs_plot, label="Training Data")

# True future data
plt.plot(range(len(inputs_plot), len(inputs_plot) + steps), targets[len(inputs_plot):len(inputs_plot) + steps], label="True Future")

# ESN Predictions
plt.plot(range(len(inputs_plot), len(inputs_plot) + steps), predictions, '--', label="ESN Predict")

plt.legend()
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.title("Echo State Network Prediction")
plt.show()
