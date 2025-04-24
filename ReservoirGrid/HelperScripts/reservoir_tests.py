import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from Models.Reservoir import Reservoir

def Test_lyapunov(Model):
    # Get reservoir state norms
    state_norms = [state.norm().item() for state in Model.reservoir_states]

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

def test_memory_capacity(Model, max_delay=50, n_trials=10):
    """
    Test the memory capacity of the reservoir by evaluating its ability
    to recall past inputs.
    """
    memory_capacities = []
    
    for _ in range(n_trials):
        # Generate random input signal
        u = np.random.uniform(-1, 1, size=(1000 + max_delay, Model.input_dim))
        
        # Train to predict delayed versions of input
        mc = 0
        for delay in range(1, max_delay + 1):
            target = u[:-delay]  # Delayed version
            Model.train_readout(u[delay:], target)
            predictions = Model.predict(u[delay:])
            
            # Calculate correlation coefficient
            corr = np.corrcoef(predictions.flatten(), target.flatten())[0,1]
            mc += corr**2
            
        memory_capacities.append(mc / max_delay)
    
    avg_mc = np.mean(memory_capacities)
    print(f"Average Memory Capacity: {avg_mc:.3f} (max possible: {max_delay})")
    return avg_mc

def test_nonlinearity(Model, n_trials=5):
    """
    Test the reservoir's ability to perform nonlinear transformations
    """
    results = []
    
    for _ in range(n_trials):
        # Generate input signal
        u = np.random.uniform(-1, 1, size=(1000, Model.input_dim))
        
        # Various nonlinear targets
        targets = {
            'Square': u**2,
            'Product': u[:,0:1] * u[:,1:2] if Model.input_dim > 1 else u**2,
            'Sin': np.sin(np.pi * u),
            'Abs': np.abs(u)
        }
        
        for name, target in targets.items():
            Model.train_readout(u, target)
            predictions = Model.predict(u)
            mse = ((predictions - target)**2).mean()
            results.append((name, mse))
    
    # Display results
    print("\nNonlinear Transformation Test Results:")
    for name, mse in results:
        print(f"{name:<10} MSE: {mse:.6f}")

def visualize_reservoir_states(Model, n_states=3):
    """
    Visualize the dynamics of individual reservoir units
    """
    if not hasattr(Model, 'reservoir_states'):
        print("No reservoir states recorded")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot first few reservoir units
    for i in range(min(n_states, Model.reservoir_dim)):
        unit_states = [state[i].item() for state in Model.reservoir_states]
        plt.plot(unit_states, label=f"Unit {i+1}")
    
    plt.title("Reservoir Unit Activations", fontsize=16)
    plt.xlabel("Time Step", fontsize=14)
    plt.ylabel("Activation", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def test_input_sensitivity(Model, n_tests=10, epsilon=1e-3):
    """
    Test how sensitive the reservoir is to small input changes
    """
    sensitivities = []
    
    for _ in range(n_tests):
        # Generate base input and slightly perturbed version
        u1 = np.random.uniform(-1, 1, size=(100, Model.input_dim))
        u2 = u1 + epsilon * np.random.randn(*u1.shape)
        
        # Get reservoir states
        states1 = Model.get_states(u1)
        states2 = Model.get_states(u2)
        
        # Calculate state differences
        diff = np.mean([(s1-s2).norm().item() for s1,s2 in zip(states1, states2)])
        sensitivity = diff / epsilon
        sensitivities.append(sensitivity)
    
    avg_sensitivity = np.mean(sensitivities)
    print(f"Average Input Sensitivity: {avg_sensitivity:.4f}")
    return avg_sensitivity

def test_temporal_processing(Model, sequence_length=20):
    """
    Test the reservoir's ability to process temporal sequences
    """
    # Generate input sequence
    u = np.zeros((1000, Model.input_dim))
    for i in range(sequence_length, len(u)):
        u[i] = 0.5 * u[i-1] + 0.3 * u[i-sequence_length] + 0.1 * np.random.randn(Model.input_dim)
    
    # Target depends on sequence history
    target = np.zeros_like(u)
    for i in range(sequence_length, len(u)):
        target[i] = np.mean(u[i-sequence_length:i], axis=0)
    
    # Train and evaluate
    Model.fit(u, target)
    predictions = Model.predict(u)
    mse = ((predictions - target)**2).mean()
    print(f"Temporal Processing MSE: {mse:.6f}")
    return mse

def spectral_analysis(Model, input_signal):
    """
    Perform spectral analysis of reservoir states
    """
    states = Model.res_states()
    state_matrix = np.array([state.numpy().flatten() for state in states])
    
    # Compute FFT for each reservoir unit
    plt.figure(figsize=(12, 6))
    for i in range(min(5, Model.reservoir_dim)):  # Plot first 5 units
        fft = np.fft.fft(state_matrix[:, i])
        freq = np.fft.fftfreq(len(state_matrix[:, i]))
        plt.plot(freq, np.abs(fft), label=f"Unit {i+1}")
    
    plt.title("Spectral Analysis of Reservoir Units", fontsize=16)
    plt.xlabel("Frequency", fontsize=14)
    plt.ylabel("Magnitude", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()