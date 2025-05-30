import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from reservoirgrid.models import Reservoir
from reservoirgrid.datasets import MackeyGlassDataset
from reservoirgrid.helpers import utils

def test_lyapunov(Model):
    # Get reservoir state norms
    state_norms = [state.norm().item() for state in Model.res_states]

    # Plot reservoir dynamics
    plt.figure(figsize=(12, 6))
    plt.plot(state_norms, label=f"Spectral Radius = {Model.spectral_radius}", color='#AC6600', linewidth=2)
    plt.xlabel("Time Step", fontsize=14, labelpad=10)
    plt.ylabel("Reservoir State Norm", fontsize=14, labelpad=10)
    plt.title("Reservoir Dynamics", fontsize=16, pad=15)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True, borderpad=1)
    plt.tight_layout()
    plt.show()
    
    # Approximate Lyapunov Exponent (Δx growth rate)
    diffs = np.abs(np.diff(state_norms))
    if len(diffs) > 0:
        lyap_exp = np.mean(np.log(diffs + 1e-6))  # Small epsilon to avoid log(0)
        print(f"Approximate Lyapunov Exponent: {lyap_exp:.4f}")
        if lyap_exp > 0:
            print("Chaotic Dynamics Detected!")
        elif lyap_exp < 0:
            print("Stable / Periodic Dynamics")
        else:
            print("Unclear Behavior")
    else:
        print("Not enough state changes to compute Lyapunov exponent.")

def lyapunov_time(pred):
    """Calculate lypunov time"""
    divergence = ((true[:len(pred)] - pred)**2).sum(axis=1)
    lyap_time = np.argmax(divergence > 0.1) if any(divergence > 0.1) else len(pred)
    return lyap_time


if __name__ == "__main__":
    
    Model = Reservoir(
    input_dim=1,
    reservoir_dim=1300,
    output_dim=1,
    spectral_radius=1,
    leak_rate=0.5,
    sparsity=0.9,
    input_scaling=0.5,
    noise_level = 0.01)
    
    Mglass1 = MackeyGlassDataset(10000, 5, tau=17, seed=0)
    inputs, targets = Mglass1[0]
    inputs, targets = utils.normalize_data(inputs), utils.normalize_data(targets)
    train_inputs, test_inputs = train_test_split(inputs, test_size = 0.2, shuffle=False)
    train_targets, test_targets = train_test_split(targets, test_size = 0.2, shuffle=False)

    Model.train_readout(train_inputs, train_targets, warmup=200)

    test_lyapunov(Model)

    predictions = Model.predict(train_inputs, steps = len(test_targets))
    print(Model.RMSE(y_true= test_targets , y_pred= predictions))
    plt.plot(predictions, color= 'green', label= "predictions")
    plt.plot(test_targets,':', color= 'blue', label= 'test_data' , )
    plt.legend()
    plt.show()
