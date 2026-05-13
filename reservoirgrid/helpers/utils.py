from time import time
from contextlib import contextmanager
from typing import Union, List, Tuple, Any, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Local imports
from reservoirgrid.models import Reservoir

@contextmanager
def timer(name):
    """Simple timing context manager"""
    start = time()
    yield
    print(f"[{name}] elapsed: {time()-start:.2f}s")

#------------------ reservoirgrid imports ---------------------#


#-------------------- Suppress UserWarning --------------------------#
import warnings

warnings.filterwarnings(
    "ignore",
    message="The following arguments have no effect for a chosen solver: `jac`.",
    category=UserWarning
)
#______________________________________________________________________#

def normalize_data(data):
    """
    Normalize data while preserving system dynamics, 
    and centers the data in the range [-1,1]
    """
    return (data - data.min()) / (data.max() - data.min()) *2 -1

def discretization_with_dt(data, length, discretization = None):
    """Generate trajectory with custom time discretization when resamplaing == false.
    resampling is false when we want varying number of points per period. This method saves computation 
    time required and gives more accurate results.
    
    Args:
        data: Dysts dataset class object 
        length: Number of points to generate
        discretization: Time step (dt) to use
        
    Returns:
        numpy.ndarray: Generated trajectory
    """
    model = data()
    if discretization is not None:
        print("discretizing manually")
        model.dt = discretization
        solution = model.make_trajectory(length, resampling = False, method= "RK45")
    return solution # type: ignore


def discretization(
    system: type,
    points_per_period_values: np.ndarray,
    trajectory_length: int,
    return_times: bool = False
) -> np.ndarray:
    """Discretize a dynamical system and return results in a structured NumPy array.
    
    Args:
        system: Class with `make_trajectory()` method.
        points_per_period_values: Array of sampling rates (must be > 0).
        trajectory_length: Number of periods to simulate.
        return_times: If True, includes time values in trajectories.

    Returns:
        Structured NumPy array with dtype=[('pp', float), ('trajectory', object)].
        Each row contains (points_per_period, trajectory_data).
    """
    # Input validation
    if not hasattr(system, 'make_trajectory'):
        raise AttributeError("System must have a 'make_trajectory' method.")
    if np.any(points_per_period_values <= 0):
        raise ValueError("All points_per_period_values must be positive.")

    # Preallocate structured array for efficiency
    results = np.empty(len(points_per_period_values), 
                      dtype=[('pp', float), ('trajectory', object)])

    for i, pp in enumerate(points_per_period_values):
        try:
            sol = system().make_trajectory(
                n=trajectory_length,
                pts_per_period=pp,
                method="RK45",
                return_times=return_times
            )
            results[i] = (pp, sol)
        except Exception as e:
            print(f"Warning: Failed for pp={pp}: {str(e)}")
            results[i] = (pp, None)  # Store None if computation fails

    return results

warnings.filterwarnings(
    "ignore",
    message="The following arguments have no effect for a chosen solver",
    category=UserWarning
)

def split(dataset:np.ndarray, window:int = 1, **kwargs):
    """
    splits dataset into training and testing sequence offsetting
    inputs and targets with a window. generally 1, but can be overwritten.
    The inputs are also converted to the torch.tensor type.
    accepts **kwargs, passed to train_test_split
    
    Args:
        dataset : input dataset, accepts dtypes accepted by sklearn
        window : offsetting parameter. Target is offsetted by window in the future
    Returns:
        train_inputs, test_inputs, train_targets, test_targets 

    """
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs, targets = dataset[:-window], dataset[window:]
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, shuffle=False, **kwargs)
    train_inputs = torch.tensor(train_inputs)
    test_inputs = torch.tensor(test_inputs)
    train_targets = torch.tensor(train_targets)
    test_targets = torch.tensor(test_targets)
    return train_inputs, test_inputs, train_targets, test_targets

def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between true and predicted values.
    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
    Returns:
        float: RMSE value.
    """
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse


def parameter_sweep(inputs, parameter_dict,
                    return_targets=True,
                    state_downsample=-1,
                    batch_size=32,  # NEW: Process 32 configs at once
                    **kwargs):
    """
    Batched parameter sweep using Reservoir_batched for GPU acceleration.
    
    Args:
        batch_size: Number of reservoir configs to train/predict simultaneously (default 32)
    """
    
    # --- 1. Data Preparation (Once) ---
    with timer("Data preparation"):
        train_inputs, test_inputs, train_targets, test_targets = split(inputs, random_state=42)
        test_targets_np = test_targets.numpy() if return_targets else None
        steps_to_predict = len(test_targets)


    # --- 2. Parameter Combinations ---
    keys_order = ["SpectralRadius", "LeakyRate", "InputScaling"]
    values = [parameter_dict[k] for k in keys_order]
    param_combinations = list(zip(*values))
    total_combinations = len(param_combinations)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move data to GPU once
    train_inputs = train_inputs.to(device, non_blocking=True)
    test_inputs = test_inputs.to(device, non_blocking=True) 
    train_targets = train_targets.to(device, non_blocking=True)
    
    results = []
    
    # --- 3. Process in Batches ---
    for batch_start in range(0, total_combinations, batch_size):
        batch_end = min(batch_start + batch_size, total_combinations)
        batch_indices = range(batch_start, batch_end)
        actual_batch_size = batch_end - batch_start
        
        print(f"\n{'='*60}")
        print(f"Batch {batch_start // batch_size + 1} | Configs {batch_start + 1}-{batch_end}/{total_combinations}")
        print(f"{'='*60}")
        
        batch_iter_start = time()
        
        try:
            # Extract parameters for this batch
            sr_batch = np.array([param_combinations[i][0] for i in batch_indices])
            lr_batch = np.array([param_combinations[i][1] for i in batch_indices])
            ins_batch = np.array([param_combinations[i][2] for i in batch_indices])
            
            # Initialize batched model ONCE per batch
            with timer(f"Batch init ({actual_batch_size} configs)"):
                batch_model = Reservoir_batched(
                    spectral_radius=sr_batch,
                    leak_rate=lr_batch,
                    input_scaling=ins_batch,
                    **{k: v for k, v in kwargs.items() if k != 'device'}
                )
                print(f"Batched model on {device}, configs={actual_batch_size}, reservoir_dim={batch_model.reservoir_dim}")
            
            # Train ALL configs at once (analytical ridge regression)
            with timer(f"Batch training ({actual_batch_size} configs)"):
                batch_model.train_readout(
                    train_inputs,
                    train_targets,
                    warmup=int(len(train_inputs) * 0.2),
                    alpha=1e-5
                )
            
            # Predict ALL configs at once
            with timer(f"Batch prediction ({actual_batch_size} configs)"):
                with torch.no_grad():
                    batch_predictions = batch_model.predict(
                        train_inputs,
                        steps=steps_to_predict
                    )  # (steps, B, O)
            
            # Extract per-config results
            with timer(f"Result extraction ({actual_batch_size} configs)"):
                for config_idx, param_idx in enumerate(batch_indices):
                    sr, lr, ins = param_combinations[param_idx]
                    
                    result = {
                        'parameters': {'SpectralRadius': sr, 'LeakyRate': lr, 'InputScaling': ins},
                        'predictions': batch_predictions[:, config_idx, :].cpu(),
                        'readout_weights': batch_model.W_out[config_idx].detach().cpu().numpy()
                    }
                    
                    if return_targets:
                        result['true_value'] = test_targets_np
                    
                    if state_downsample > 0:
                        # Extract states for this config only
                        result['reservoir_states'] = batch_model.reservoir_states[:, config_idx, :].detach().cpu().numpy()[::state_downsample]
                    
                    results.append(result)
            
            batch_iter_end = time()
            print(f"Batch time: {batch_iter_end - batch_iter_start:.2f}s ({actual_batch_size} configs)")
            
        except Exception as e:
            print(f"Error in batch {batch_start // batch_size + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        finally:
            # Cleanup batch model
            if 'batch_model' in locals():
                del batch_model # type: ignore
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"Total combinations processed: {len(results)}/{total_combinations}")
    print(f"{'='*60}\n")
    
    return results

def truncate(system):
    """
    trancate the system to the length of least period sample of the system. 

    Args:
        system: Accepts an ensamble of the same system but with different point per period.
    returns:
        system: The system with the equal number of period instead of points.
    """
    PP_array = system['pp'] # Return system's points per period in a list

    l_period = len(system['trajectory'][-1])//PP_array[-1] # Length of the fewest period dataset for referance to make all sample of this exact period

    for i in range(len(PP_array)):
        num_points = int(l_period * PP_array[i]) # Calculates the points required for l_periods
        system['trajectory'][i]= system['trajectory'][i][:num_points] # Slices till the points required reached

    return system
