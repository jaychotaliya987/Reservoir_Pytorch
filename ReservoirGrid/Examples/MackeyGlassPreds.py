import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Ensure the correct path to MackeyGlass module
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from Datasets.MackeyGlassDataset import MackeyGlassDataset
from Models.Reservoir import Reservoir
from Models.Echostate import ESN 

print("Imports Done!\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

Mglass1 = MackeyGlassDataset(10000, 5, tau=17, seed=0)
inputs, targets = Mglass1[0]
inputs = inputs.to(device)
targets = targets.to(device)

# Create reservoir
reservoir = Reservoir(input_dim=1, reservoir_dim=500, output_dim=1, 
                     spectral_radius=1.1, leak_rate=0.3, sparsity=0.9)
reservoir = reservoir.to(device)

# Train the readout layer
reservoir.train_readout(inputs, targets, alpha=1e-6)

# Predictions
steps = 1000
with torch.no_grad():
    initial_input = inputs[-1:] 
    predictions = reservoir.predict(initial_input, steps=steps, teacher_forcing=inputs, warmup=3)
predictions = predictions.squeeze(1).cpu().numpy()

inputs_plot = inputs[:-steps].squeeze(1).cpu().numpy()  # (800,)
true_future = targets[-steps:].squeeze(1).cpu().numpy()  # (200,)

######-------------------Plots-------------------######
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6), dpi=100)

# Training data
plt.plot(range(len(inputs_plot)), inputs_plot, 
         color='#1f77b4', linewidth=2.5, 
         label="Training Data")

# True future data
plt.plot(range(len(inputs_plot), len(inputs_plot) + steps), 
         true_future, 
         color='#2ca02c', linewidth=2.5,
         label="True Future")

# Reservoir Predictions
plt.plot(range(len(inputs_plot), len(inputs_plot) + steps), 
         predictions, 
         color='#ff7f0e', linewidth=2,
         marker='o', markersize=2, markevery=5,
         label="Reservoir Predictions")

# Vertical line
plt.axvline(x=len(inputs_plot), color='gray', linestyle=':', alpha=0.7)

# Annotations
plt.annotate('Prediction Start', 
             xy=(len(inputs_plot), np.min(inputs_plot)), 
             xytext=(10, 10), textcoords='offset points',
             arrowprops=dict(arrowstyle="->"))

# Formatting
plt.title("Mackey-Glass Time Series Prediction", fontsize=16, pad=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0, len(inputs_plot) + steps)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("System State", fontsize=12)
plt.legend(loc='upper right', framealpha=1)

# Adjust layout
plt.tight_layout()
plt.show()