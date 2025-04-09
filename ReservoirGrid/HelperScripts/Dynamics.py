import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from Models.Reservoir import Reservoir

# Function to test ESN with different spectral radii
def test_esn(spectral_radius=0.9, leak_rate=0.2, sparsity=0.9, W_in_scale=0.1, steps=500):
    input_dim, reservoir_dim, output_dim = 1, 100, 1
    
    # Initialize ESN with given parameters
    esn = Reservoir(input_dim, reservoir_dim, output_dim, spectral_radius=spectral_radius, leak_rate=leak_rate)
    
    # Generate a random input sequence
    torch.manual_seed(42)
    inputs = torch.randn(steps, input_dim) * W_in_scale
    
    # Run ESN forward
    with torch.no_grad():
        _ = esn(inputs, reset_state=True)
    
    # Get reservoir state norms
    state_norms = [state.norm().item() for state in esn.reservoir_states]

    # Plot reservoir dynamics
    plt.figure(figsize=(12, 6))
    plt.plot(state_norms, label=f"Spectral Radius = {spectral_radius}", color='#AC6600', linewidth=2)
    plt.xlabel("Time Step", fontsize=14, labelpad=10)
    plt.ylabel("Reservoir State Norm", fontsize=14, labelpad=10)
    plt.title("Reservoir Dynamics", fontsize=16, pad=15)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True, borderpad=1)
    plt.tight_layout()
    plt.savefig(f"test_{spectral_radius}.png",dpi=600)
    
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
            print("⚠ Unclear Behavior")
    else:
        print("⚠ Not enough state changes to compute Lyapunov exponent.")

