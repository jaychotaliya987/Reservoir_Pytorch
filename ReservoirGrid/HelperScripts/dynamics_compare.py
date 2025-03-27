import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Function to test ESN with multiple spectral radii
def test_esn_compare_radii(spectral_radii=[0.5, 0.9, 1.2], leak_rate=0.2, sparsity=0.9, W_in_scale=0.1, steps=500):
    input_dim, reservoir_dim, output_dim = 1, 100, 1
    
    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, spectral_radius in enumerate(spectral_radii):
        # Initialize ESN
        esn = Reservoir(input_dim, reservoir_dim, output_dim, 
                        spectral_radius=spectral_radius, leak_rate=leak_rate)
        
        # Generate random input
        torch.manual_seed(42)
        inputs = torch.randn(steps, input_dim) * W_in_scale
        
        # Run ESN forward
        with torch.no_grad():
            _ = esn(inputs, reset_state=True)
        
        # Get state norms and Lyapunov exponent
        state_norms = [state.norm().item() for state in esn.reservoir_states]
        diffs = np.abs(np.diff(state_norms))
        lyap_exp = np.mean(np.log(diffs + 1e-6)) if len(diffs) > 0 else np.nan
        
        # Plot dynamics
        plt.plot(state_norms, 
                 label=f"ρ = {spectral_radius} (λ ≈ {lyap_exp:.2f})", 
                 color=colors[i], 
                 linewidth=2)
        
    # Plot formatting
    plt.xlabel("Time Step", fontsize=14)
    plt.ylabel("Reservoir State Norm", fontsize=14)
    plt.title(f"Reservoir Dynamics (Leak Rate = {leak_rate})", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, framealpha=1, shadow=True)
    plt.tight_layout()
    plt.savefig("reservoir_dynamics_comparison.png", dpi=600)
    plt.show()

