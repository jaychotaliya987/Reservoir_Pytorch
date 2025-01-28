import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import numpy 
import matplotlib.pyplot as plt

import sys
import os

# Ensure the correct path to MackeyGlass module
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from Datasets.MackeyGlassDataset import MackeyGlassDataset
from Models.Echostate import ESN

print("Imports Done!\n")

## INITIALIZATION
Mglass1 = MackeyGlassDataset(100, 5, tau=20, seed=0)
esn = ESN(input_dim=1, reservoir_dim=200, output_dim=1)
epochs = 100

## DATA PREPARATION
Train_test_Split = 0.8
train_size = int(Train_test_Split * len(Mglass1))
test_size = len(Mglass1) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(Mglass1, [train_size, test_size])

# Extract data from the dataset for plotting
train_data = [data[0].numpy() for data in train_dataset]
train_data = numpy.concatenate(train_data, axis=0)

plt.plot(train_data)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

## TRAINING
esn = esn.to(device)  # Move model to GPU
esn.freeze_reservoir()  # Freeze reservoir weights

# Define optimizer and criterion for the readout layer
optimizer = torch.optim.Adam(esn.readout.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Convert NumPy arrays to tensors and move to GPU
inputs = torch.from_numpy(train_data).float().to(device)
targets = torch.from_numpy(train_data).float().to(device)

# Debug device placement
print(f"Model is on: {next(esn.parameters()).device}")
print(f"Inputs are on: {inputs.device}")
print(f"Targets are on: {targets.device}")

esn.Train(dataset= inputs, epochs=30, lr=0.0001, 
          criterion=nn.MSELoss, print_every=5)
