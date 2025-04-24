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
from _datasets.MackeyGlassDataset import MackeyGlassDataset
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

######-------------------Plots-------------------######
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style and palette
sns.set_style("white")
sns.set_palette(['#8da0cb', '#66c2a5', '#fc8d62'])  # custom color set
sns.despine()

plt.figure(figsize=(12, 6), dpi=200)

# Plot training data
inputs_plot = inputs.squeeze(1).cpu().numpy()
plt.plot(range(len(inputs_plot)), inputs_plot,
         color='#8da0cb', linewidth=2.5,
         label="Training Data")

# Plot true future
true_future = inputs[-steps:].squeeze(1).cpu().numpy()
plt.plot(range(len(inputs_plot), len(inputs_plot) + steps),
         true_future,
         color='#66c2a5', linewidth=2.5,
         label="True Future")

# Plot predictions
plt.plot(range(len(inputs_plot), len(inputs_plot) + steps),
         predictions,
         color='#fc8d62', linewidth=2.5,
         linestyle='--', alpha=0.8,
         label="Reservoir Predictions")

# Vertical line at prediction start
plt.axvline(x=len(inputs_plot), color='gray', linestyle=':', linewidth=1.2, alpha=0.7)

# Annotate prediction start
plt.annotate('Prediction Start',
             xy=(len(inputs_plot), true_future[0]),
             xytext=(len(inputs_plot)-60, true_future[0]+0.05),
             textcoords='data',
             arrowprops=dict(arrowstyle="->", color='gray'),
             fontsize=12)

# Labels and title
plt.title("Mackey-Glass Time Series Forecast", fontsize=18, pad=15, weight='semibold')
plt.xlabel("Time Step", fontsize=13)
plt.ylabel("System State", fontsize=13)

# Ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Limits
plt.xlim(0, len(inputs_plot) + steps)

# Legend
plt.legend(loc='upper right', fontsize=12)

# Layout and save (optional)
plt.tight_layout()
# plt.savefig("forecast_plot.png", dpi=300, bbox_inches='tight')

plt.show()
