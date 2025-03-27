import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
    plt.figure(figsize=(10, 5))
    plt.plot(state_norms, label=f"Spectral Radius = {spectral_radius}")
    plt.xlabel("Time Step")
    plt.ylabel("Reservoir State Norm")
    plt.title("Reservoir Dynamics")
    plt.legend()
    plt.show()
    
    # Approximate Lyapunov Exponent (Δx growth rate)
    diffs = np.abs(np.diff(state_norms))
    if len(diffs) > 0:
        lyap_exp = np.mean(np.log(diffs + 1e-6))  # Small epsilon to avoid log(0)
        print(f"Approximate Lyapunov Exponent: {lyap_exp:.4f}")
        if lyap_exp > 0:
            print("🔴 Chaotic Dynamics Detected!")
        elif lyap_exp < 0:
            print("🟢 Stable / Periodic Dynamics")
        else:
            print("⚠ Unclear Behavior")
    else:
        print("⚠ Not enough state changes to compute Lyapunov exponent.")

# Run tests with different spectral radii
test_esn(spectral_radius=0.7)  # Low: likely periodic
test_esn(spectral_radius=0.9)  # Recommended: should be near edge of chaos
test_esn(spectral_radius=1.3)  # High: might be chaotic
