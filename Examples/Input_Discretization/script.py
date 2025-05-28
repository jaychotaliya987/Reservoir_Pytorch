from timeit import default_timer as timer

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from reservoirgrid.models import Reservoir
from reservoirgrid.datasets import LorenzAttractor
#from dysts.datasets import load_dataset
import pandas as pd

def normalize_data(data):
    """Normalize data while preserving Lorenz system dynamics"""
    return (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

def prepare_data(data, test_ratio=0.2):
    """Sequential split for time series data"""
    n = len(data)
    split_idx = int(n * (1 - test_ratio))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data

def create_reservoir(reservoir_size=1000, spectral_radius=1.25, leak_rate=0.3, sparsity=0.5, input_scaling=0.5):
    """Create reservoir with specified parameters"""
    return Reservoir(
        input_dim=3,
        reservoir_dim=reservoir_size,
        output_dim=3,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        sparsity=sparsity,
        input_scaling=input_scaling,
    )

def train_and_predict(reservoir_params, warmup=500, prediction_steps=200):
    """Complete training and prediction pipeline"""
    # Generate and prepare data
    attractor = np.genfromtxt('reservoirgrid/datasets/Lorenz_fine.csv', delimiter=",", skip_header=1)
    #data = np.genfromtxt("data.csv", delimiter=",", skip_header=1)
    attractor = torch.tensor(attractor[:,1:])
    attractor = normalize_data(attractor)
    input, target = attractor[:-1], attractor[1:]
    input_train, input_test = prepare_data(input)
    target_train, target_test = prepare_data(target)
    
    # Create and train reservoir
    reservoir = create_reservoir(**reservoir_params)
    reservoir.train_readout(input_train, target_train, warmup=warmup)
    
    # Predict in multiple shorter segments
    predictions = []
    current_state = input_test[0:1]
    for _ in range(prediction_steps):
        pred = reservoir.predict(current_state, steps=1)
        predictions.append(pred)
        current_state = pred  # Autoregressive prediction
    
    predictions = torch.cat(predictions)
    return predictions, input_test, target_test

def evaluate(true, pred):
    """Calculate evaluation metrics"""
    true = true.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    mse = mean_squared_error(true[:len(pred)], pred)
    divergence = ((true[:len(pred)] - pred)**2).sum(axis=1)
    lyap_time = np.argmax(divergence > 0.1) if any(divergence > 0.1) else len(pred)
    return mse, lyap_time

def plot_results(true, pred, title=""):
    """Visualize results in 3D and time series"""
    fig = plt.figure(figsize=(18, 10))
    
    # 3D Trajectory
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot(true[:,0], true[:,1], true[:,2], lw=0.5, label='True', alpha=0.7)
    ax1.plot(pred[:,0], pred[:,1], pred[:,2], lw=0.5, label='Predicted', alpha=0.7)
    ax1.set_title(f"{title}\n3D Trajectory")
    ax1.legend()
    
    # Time series components
    time_steps = min(200, len(true), len(pred))
    for i, comp in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(234+i)
        ax.plot(true[:time_steps,i], label=f'True {comp}')
        ax.plot(pred[:time_steps,i], label=f'Pred {comp}')
        ax.set_title(f"{comp} Component")
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def parameter_sweep():
    """Test different reservoir configurations"""
    param_combinations = [
    {'reservoir_size': 1000, 'spectral_radius': 0.7, 'leak_rate': 0.5, 'sparsity': 0.9, 'input_scaling': 0.5},
    {'reservoir_size': 1000, 'spectral_radius': 0.9, 'leak_rate': 0.5, 'sparsity': 0.9, 'input_scaling': 0.5},
    {'reservoir_size': 1000, 'spectral_radius': 1, 'leak_rate': 0.5, 'sparsity': 0.9, 'input_scaling': 0.5},
    {'reservoir_size': 1000, 'spectral_radius': 1.2, 'leak_rate': 0.5, 'sparsity': 0.9, 'input_scaling': 0.5}
    ]
    
    results = []
    for params in param_combinations:
        start_time = timer()
        predictions, input_test, target_test = train_and_predict(params)
        mse, lyap_time = evaluate(target_test[:200], predictions[:200])
        results.append({
            'params': params,
            'mse': mse,
            'lyap_time': lyap_time,
            'time': timer() - start_time
        })
        target_test = target_test.detach().cpu()
        predictions = predictions.detach().cpu()
        plot_results(target_test.numpy()[:200], predictions.numpy()[:200], 
                    title=f"Config: {params}")
    
    # Print results summary
    print("\n=== Results Summary ===")
    for i, res in enumerate(results, 1):
        print(f"\nConfiguration {i}:")
        print(f"Parameters: {res['params']}")
        print(f"MSE: {res['mse']:.4f}")
        print(f"Lyapunov time: {res['lyap_time']} steps")
        print(f"Runtime: {res['time']:.2f} seconds")

if __name__ == "__main__":
    # Run the parameter sweep
    #parameter_sweep()
    
#    # Example single run with best parameters
#    print("\nRunning best configuration...")
#    best_params = {'reservoir_size': 1000, 'spectral_radius': , 'leak_rate': 0.2}
#    predictions, input_test, target_test = train_and_predict(best_params, prediction_steps=500)
#    mse, lyap_time = evaluate(target_test[:500], predictions[:500])
#    print(f"\nBest configuration results:")
#    print(f"MSE: {mse:.4f}")
#    print(f"Lyapunov time: {lyap_time} steps")
#    plot_results(target_test.numpy()[:500], predictions.numpy()[:500], 
#                title=f"Best Configuration: {best_params}")
    