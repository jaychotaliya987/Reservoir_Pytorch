import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

# Ensure the correct path to MackeyGlass module
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from Datasets.MackeyGlassDataset import MackeyGlassDataset
from Models.Reservoir import Reservoir
from Models.Echostate import ESN

print("Imports Done!\n")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Mglass1 = MackeyGlassDataset(10000, 5, tau=17, seed=0)
inputs, targets = Mglass1[0]
inputs = inputs.to(device)
targets = targets.to(device)

# Create reservoir
reservoir = Reservoir(input_dim=1, reservoir_dim=500, output_dim=1, 
                     spectral_radius=0.90, leak_rate=0.3, sparsity=0.9)
reservoir = reservoir.to(device)

reservoir2 = ESN(input_dim=1, reservoir_dim=500, output_dim=1,
                  spectral_radius=0.90, leak_rate=0.3, sparsity=0.9)
reservoir2 = reservoir2.to(device)


# Train the readout layer
reservoir.train_readout(inputs, targets, alpha=1e-6)
reservoir2.train_readout(inputs, targets, alpha=1e-6)

# Predict future values
steps = 1000
with torch.no_grad():
    initial_input = inputs[-1:] 
    predictions = reservoir.predict(initial_input, steps=steps, teacher_forcing=None, warmup=0)
    predictions2 = reservoir2.predict(initial_input, steps=steps, teacher_forcing=None, warmup=0)
predictions = predictions.squeeze(1).cpu().numpy()
predictions2 = predictions2.squeeze(1).cpu().numpy()


inputs_plot = inputs[:-steps].squeeze(1).cpu().numpy()  # (800,)
true_future = targets[-steps:].squeeze(1).cpu().numpy()  # (200,)

# Set up the plot
plt.figure(figsize=(12, 6), dpi=100)
plt.grid(True, alpha=0.3)
plt.gca().set_facecolor('#f8f9fa')  # Light gray background

# Training data (thicker line)
plt.plot(range(len(inputs_plot)), inputs_plot, 
         color='#1f77b4', linewidth=2.5, 
         label="Training Data")

# True future data (solid line)
plt.plot(range(len(inputs_plot), len(inputs_plot) + steps), 
         true_future, 
         color='#2ca02c', linewidth=2.5,
         label="True Future")

# Reservoir Predictions (dashed with markers)
plt.plot(range(len(inputs_plot), len(inputs_plot) + steps), 
         predictions, 
         color='#ff7f0e', linewidth=2,
         marker='o', markersize=2, markevery=5,
         label="Reservoir Predictions")

# Add vertical line at prediction start
plt.axvline(x=len(inputs_plot), color='gray', linestyle=':', alpha=0.7)

# Add annotations
plt.annotate('Prediction Start', 
             xy=(len(inputs_plot), np.min(inputs_plot)), 
             xytext=(10, 10), textcoords='offset points',
             arrowprops=dict(arrowstyle="->"))

# Formatting
plt.title("Mackey-Glass Time Series Prediction\nReservoir Computing Performance", 
          fontsize=14, pad=20)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("System State", fontsize=12)
plt.legend(loc='upper right', framealpha=1)

# Adjust layout
plt.tight_layout()
plt.show()