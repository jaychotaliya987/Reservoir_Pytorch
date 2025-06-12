import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Union

def test_memory_capacity(Model, max_delay: int = 50, n_trials: int = 10) -> float:
    """
    Tests the reservoir's memory capacity by evaluating its ability to recall past inputs.
    
    The memory capacity is calculated by measuring how well the reservoir can reconstruct
    delayed versions of its input signal, averaged across multiple delays and trials.
    
    Args:
        Model: The reservoir model instance. Must implement:
               - device(): returns current device (cpu/cuda)
               - input_dim: input dimension
               - train_readout(): trains readout weights
               - predict(): generates predictions
        max_delay: Maximum time delay to test (in timesteps). Default 50.
        n_trials: Number of independent trials to average over. Default 10.
    
    Returns:
        float: Average memory capacity score between 0 and max_delay
        
    Interpretation:
        - Scores close to max_delay indicate excellent memory retention
        - Scores <10 suggest poor short-term memory
        - Typical values for good reservoirs: 20-40 for max_delay=50
        - Compare against theoretical maximum (max_delay) for relative performance
    """
    device = Model.device
    memory_capacities = []
    delay_correlations = []

    for _ in range(n_trials):
        u = torch.rand(1000 + max_delay, Model.input_dim, device=device) * 2 - 1
        
        mc = 0.0
        trial_corrs = []
        for delay in range(1, max_delay + 1):
            target = u[:-delay]
            Model.train_readout(u[delay:], target)
            predictions = Model.predict(u[delay:], steps=len(target))
            
            cov = torch.cov(torch.stack([predictions.flatten(), target.flatten()]))
            corr = cov[0,1] / (torch.sqrt(cov[0,0] * cov[1,1]) + 1e-8)
            mc += corr**2
            trial_corrs.append(corr.item())
        
        memory_capacities.append(mc.item() / max_delay)
        delay_correlations.append(trial_corrs)

    # Plot correlation vs delay
    fig = go.Figure()
    avg_corrs = np.mean(delay_correlations, axis=0)
    
    fig.add_trace(go.Scatter(
        x=list(range(1, max_delay+1)),
        y=avg_corrs,
        mode='lines+markers',
        name='Correlation'
    ))
    
    fig.update_layout(
        title=f'Memory Capacity: Avg {np.mean(memory_capacities):.2f}/{max_delay}',
        xaxis_title='Delay (steps)',
        yaxis_title='Correlation',
        hovermode='x'
    )
    fig.show()

    return np.mean(memory_capacities)

def test_nonlinearity(Model, n_trials: int = 5) -> List[Tuple[str, float]]:
    """
    Evaluates the reservoir's ability to perform nonlinear transformations.
    
    Tests performance on four fundamental nonlinear operations:
    - Squaring
    - Multiplication (if input_dim > 1)
    - Sine function
    - Absolute value
    
    Args:
        Model: Reservoir model instance with same requirements as test_memory_capacity
        n_trials: Number of independent test runs. Default 5.
    
    Returns:
        List[Tuple[str, float]]: List of (operation_name, MSE) pairs
        
    Interpretation:
        - Lower MSE indicates better nonlinear processing capability
        - Compare across operations to identify strengths/weaknesses
        - Good reservoirs typically achieve MSE < 0.05 for basic nonlinearities
        - Sine function is usually most challenging
    """
    device = Model.device
    results = []
    
    for _ in range(n_trials):
        u = torch.rand(1000, Model.input_dim, device=device) * 2 - 1
        
        targets = {
            'Square': u**2,
            'Product': u[:,0:1] * u[:,1:2] if Model.input_dim > 1 else u**2,
            'Sin': torch.sin(np.pi * u),
            'Abs': torch.abs(u)
        }
        
        for name, target in targets.items():
            Model.train_readout(u, target)
            predictions = Model.predict(u)
            mse = torch.mean((predictions - target)**2).item()
            results.append({'Operation': name, 'MSE': mse, 'Trial': _+1})
    
    # Create parallel coordinates plot
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(color='blue'),
            dimensions=list([
                dict(label='Operation', values=[r['Operation'] for r in results]),
                dict(label='MSE', values=[r['MSE'] for r in results]),
                dict(label='Trial', values=[r['Trial'] for r in results])
            ])
        )
    )
    
    fig.update_layout(
        title='Nonlinear Transformation Performance',
        height=400
    )
    fig.show()

    return results

def visualize_reservoir_states(Model, n_states: int = 3) -> None:
    """
    Visualizes the temporal dynamics of reservoir units.
    
    Plots the activation trajectories of individual reservoir units over time,
    helping to analyze the diversity and stability of internal representations.
    
    Args:
        Model: Reservoir model that has recorded states in 'reservoir_states' attribute
        n_states: Number of units to visualize. Default 3.
    
    Interpretation:
        - Healthy reservoirs show diverse, non-saturated activation patterns
        - Chaotic oscillations may indicate instability
        - Flat lines suggest dead units
        - Ideal: Mix of periodic and chaotic behaviors with amplitudes in [-1,1] range
    """
    if not hasattr(Model, 'reservoir_states'):
        print("No reservoir states recorded")
        return
    
    states_cpu = [state.cpu() for state in Model.reservoir_states]
    time_steps = list(range(len(states_cpu)))
    
    fig = go.Figure()
    
    for i in range(min(n_states, Model.reservoir_dim)):
        unit_states = [state[i].item() for state in states_cpu]
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=unit_states,
            mode='lines',
            name=f'Unit {i+1}',
            hovertemplate='Time: %{x}<br>Activation: %{y:.4f}'
        ))
    
    fig.update_layout(
        title='Reservoir Unit Activations',
        xaxis_title='Time Step',
        yaxis_title='Activation',
        hovermode='x unified',
        height=500
    )
    fig.show()

def test_input_sensitivity(Model, n_tests: int = 10, epsilon: float = 1e-3) -> float:
    """
    Quantifies how sensitive the reservoir is to small input perturbations.
    
    Measures the average normalized difference in reservoir states caused by
    small random perturbations to the input signal.
    
    Args:
        Model: Reservoir model with get_states() method
        n_tests: Number of test cases. Default 10.
        epsilon: Perturbation magnitude. Default 1e-3.
    
    Returns:
        float: Average sensitivity metric
        
    Interpretation:
        - Values between 1-10 indicate healthy sensitivity
        - <1 suggests overly damped responses
        - >100 may indicate chaotic instability
        - Ideal depends on application: 
          - 2-5 for robust pattern recognition
          - 5-10 for time-series forecasting
    """
    device = Model.device
    sensitivities = []
    perturbation_data = []
    
    for _ in range(n_tests):
        u1 = torch.rand(100, Model.input_dim, device=device) * 2 - 1
        u2 = u1 + epsilon * torch.randn_like(u1)
        
        states1 = Model.get_states(u1)
        states2 = Model.get_states(u2)
        
        diff = torch.mean(torch.stack([torch.norm(s1-s2) 
                         for s1,s2 in zip(states1, states2)])).item()
        sensitivity = diff / epsilon
        sensitivities.append(sensitivity)
        
        # Store sample perturbation data for visualization
        if _ == 0:
            sample_diff = [torch.norm(s1-s2).item() for s1,s2 in zip(states1, states2)]
            perturbation_data = sample_diff
    
    # Plot sensitivity example
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=perturbation_data,
        mode='lines',
        name='State Difference Norm',
        hovertemplate='Step: %{x}<br>ΔState: %{y:.4f}'
    ))
    
    fig.update_layout(
        title=f'Input Sensitivity Example (Avg: {np.mean(sensitivities):.2f})',
        xaxis_title='Time Step',
        yaxis_title='||State Difference||',
        height=400
    )
    fig.show()

    return np.mean(sensitivities)

def spectral_analysis(Model, input_signal: torch.Tensor) -> None:
    """
    
    Performs frequency domain analysis of reservoir unit activations.
    
    Computes and plots the FFT magnitude spectra of reservoir states
    to analyze the frequency characteristics of the dynamics.
    
    Args:
        Model: Reservoir model with res_states() method
        input_signal: Example input signal to drive the reservoir
    
    Interpretation:
        - Broad spectra indicate rich dynamics
        - Sharp peaks suggest periodic behavior
        - High low-frequency power may indicate slow features
        - Ideal: Balanced distribution across frequencies
    
    """
    states = Model.res_states
    state_matrix = torch.stack([state.flatten() for state in states]).cpu().numpy()
    
    fig = go.Figure()
    
    for i in range(min(5, Model.reservoir_dim)):
        fft = np.fft.fft(state_matrix[:, i])
        freq = np.fft.fftfreq(len(state_matrix[:, i]))
        
        # Plot positive frequencies only
        mask = freq > 0
        fig.add_trace(go.Scatter(
            x=freq[mask],
            y=np.abs(fft[mask]),
            mode='lines',
            name=f'Unit {i+1}',
            hovertemplate='Freq: %{x:.4f}<br>Magnitude: %{y:.4f}'
        ))
    
    fig.update_layout(
        title='Spectral Analysis of Reservoir Units',
        xaxis_title='Frequency',
        yaxis_title='Magnitude',
        hovermode='x unified',
        height=500
    )
    fig.show()

def test_temporal_processing(Model, sequence_length: int = 20) -> float:
    """
    Evaluates the reservoir's ability to process temporal sequences.
    
    Tests performance on a task requiring integration of information over time,
    where the target depends on a history of previous inputs.
    
    Args:
        Model: Reservoir model with fit() and predict() methods
        sequence_length: Time window for temporal dependencies. Default 20.
    
    Returns:
        float: Mean squared error on the temporal task
        
    Interpretation:
        - MSE < 0.01 indicates excellent temporal processing
        - MSE 0.01-0.05 is acceptable for many applications
        - MSE > 0.1 suggests poor temporal integration
        - Compare with memory capacity results for full analysis
    """
    device = Model.device
    u = torch.zeros((1000, Model.input_dim), device=device)
    
    for i in range(sequence_length, len(u)):
        u[i] = 0.5 * u[i-1] + 0.3 * u[i-sequence_length] + 0.1 * torch.randn(Model.input_dim, device=device)
    
    target = torch.zeros_like(u)
    for i in range(sequence_length, len(u)):
        target[i] = torch.mean(u[i-sequence_length:i], dim=0)
    
    Model.fit(u, target)
    predictions = Model.predict(u)
    mse = torch.mean((predictions - target)**2).item()
    print(f"Temporal Processing MSE: {mse:.6f}")
    return mse


def best_hyperparam():
    pass