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
matplotlib.use('qt5Agg')  # Use a non-interactive backend for saving plots

# Ensure the correct path to MackeyGlass module
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from Datasets.MackeyGlassDataset import MackeyGlassDataset
from Models.Reservoir import Reservoir

print("Imports Done!\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

Mglass1 = MackeyGlassDataset(10000, 5, tau=17, seed=0)
inputs, targets = Mglass1[0]
inputs = inputs.to(device)
targets = targets.to(device)

# Create reservoir
reservoir = Reservoir(input_dim=1, reservoir_dim=2000, output_dim=1, 
                     spectral_radius=1.2, leak_rate=0.3, sparsity=0.95)
reservoir = reservoir.to(device)

steps = 1000

# Train the readout layer
reservoir.train_readout(inputs[:-steps], targets[:-steps], alpha=1e-6)

# Predictions

predictions = reservoir.predict(inputs, steps=steps, teacher_forcing=None, warmup=0)
predictions = predictions.squeeze(1).cpu().numpy()

print(f"Predictions shape: {predictions.shape}")
print(f"Inputs shape: {inputs.shape}")
print(f"Targets shape: {targets.shape}")
print(inputs[-1] == targets[-2])


######-------------------Plots-------------------######
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6), dpi=100)

# Training data (plot FULL input sequence)
inputs_plot = inputs.squeeze(1).cpu().numpy()  # (1000,)
plt.plot(range(len(inputs_plot)), inputs_plot, 
         color='#1f77b4', linewidth=2.5, 
         label="Training Data")

# True future data (aligned with predictions)
true_future = inputs[-steps:].squeeze(1).cpu().numpy()  # (200,)
plt.plot(range(len(inputs_plot), len(inputs_plot) + steps), 
         true_future, 
         color='#2ca02c', linewidth=2.5,
         label="True Future")

# Reservoir Predictions (start right after inputs end)
plt.plot(range(len(inputs_plot), len(inputs_plot) + steps), 
         predictions, 
         color='#ff7f0e', linewidth=2,
         marker='o', markersize=2, markevery=5,
         label="Reservoir Predictions")

# Vertical line at prediction start
plt.axvline(x=len(inputs_plot), color='gray', linestyle=':')

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

plt.tight_layout()
plt.show()