import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
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
Mglass1 = MackeyGlassDataset(1000, 5, tau=20, seed=0)

inputs, targets = Mglass1[0]
inputs = inputs.to(device) 
targets = targets.to(device)

esn = ESN(input_dim=1, reservoir_dim=200, output_dim=1, spectral_radius=0.95)
esn = esn.to(device)  

# Debug device placement
print(f"Model is on: {next(esn.parameters()).device}")
print(f"Inputs are on: {inputs.device}")
print(f"Targets are on: {targets.device}")

# Train the model
esn.freeze_reservoir()
losses = esn.Train(dataset=inputs, targets=targets, epochs=1000, 
                   lr=0.001, criterion=nn.MSELoss, print_every=10)

# Plot training losses
plt.plot(losses.cpu().detach().numpy())
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()

## PREDICTION
# Predict future values
future_steps = 200
predictions = esn.Predict(inputs, future_steps)

plt.plot(targets[:200].cpu().detach().numpy(), label="True")
plt.plot(predictions.cpu().detach().numpy(), label="Predicted")
plt.title("Mackey-Glass Prediction")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
