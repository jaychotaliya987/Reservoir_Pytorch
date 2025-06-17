from timeit import default_timer as timer
import gc
import tracemalloc

import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import Union, List, Tuple, Any
from itertools import product

import torch
from sklearn.model_selection import train_test_split
import numpy as np

from dysts.flows import *


from time import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    """Simple timing context manager"""
    start = time()
    yield
    print(f"[{name}] elapsed: {time()-start:.2f}s")

#------------------ reservoirgrid imports ---------------------#
from reservoirgrid.models import Reservoir


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
    return solution


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs, targets = dataset[:-window], dataset[window:]
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, shuffle=False, **kwargs)
    train_inputs = torch.tensor(train_inputs).to(device)
    test_inputs = torch.tensor(test_inputs).to(device)
    train_targets = torch.tensor(train_targets).to(device)
    test_targets = torch.tensor(test_targets).to(device)
    return train_inputs, test_inputs, train_targets, test_targets

def RMSE(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between true and predicted values.
    Args:
        y_true (torch.Tensor): True values.
        y_pred (torch.Tensor): Predicted values.
    Returns:
        float: RMSE value.
    """
    
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    return rmse.item()

def parameter_sweep(inputs, parameter_dict, 
                    return_targets=False, 
                    state_downsample=10,
                    **kwargs):
                    
    """
    Generates the reservoir, train the readout with Ridge Regression and generates the predictions on the system.
    splits the data for RMSE and have a option to return the test sequence and predictions for furthur use.

    Args:
        inputs: This is a plain input sequence that of type numpy.ndarray
        parameter_dict : This is a dictionary of parameters to sweep through. This only accepts 3 main parameter of the RC
                        1. Spectral Radius, 2.Leaky Rate, 3. Input Scaling in that order.
        **kwargs : This are all the parameters passed to the model.Reservoir class for generation. Intrinsically need all the parameters 
                    needed for the generation.
    returns: 
        results: A dictionary of the parameters with the prediction. Optionally with the test sequance.

    """
    # Pre-process
    with timer("Data preparation"):
        train_inputs, test_inputs, train_targets, test_targets = split(inputs, random_state=42)
        test_targets = test_targets.detach().cpu()
        test_targets_np = test_targets.numpy() if return_targets else None
        steps_to_predict = len(test_targets)
        
        # Convert to tuples to avoid repeated dict lookups
        sr_values = tuple(parameter_dict["SpectralRadius"])
        lr_values = tuple(parameter_dict["LeakyRate"])
        ins_values = tuple(parameter_dict["InputScaling"])
        param_combi = product(sr_values, lr_values, ins_values)
        total_combinations = len(sr_values) * len(lr_values) * len(ins_values)

    results = []
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    for i, (sr, lr, ins) in enumerate(param_combi, 1):
        iter_start = time()
        print(f"\nCombination {i}/{total_combinations} - SR: {sr}, LR: {lr}, IS: {ins}")
        
        try:
            # Memory cleanup
            torch.cuda.empty_cache() if 'cuda' in device else None
            
            # Model initialization
            with timer("Model init"):
                model = Reservoir(
                    spectral_radius=sr,
                    leak_rate=lr,
                    input_scaling=ins,
                    **{k:v for k,v in kwargs.items() if k != 'device'}
                )
            
            # Training
            with timer("Training"):
                model.train_readout(
                    train_inputs,
                    train_targets,
                    warmup=int(len(train_inputs)*0.2),
                    alpha=1e-5  # Ridge parameter
                )
            
            # Prediction
            with timer("Prediction"):
                with torch.no_grad():
                    prediction = model.predict(
                        train_inputs, 
                        steps=steps_to_predict
                    ).cpu()
                    rmse = RMSE(test_targets, prediction)
            
            # Store results (memory efficiently)
            result = {
                'parameters': {'SpectralRadius': sr, 'LeakyRate': lr, 'InputScaling': ins},
                'metrics': {'RMSE': float(rmse)},  # Convert to Python float
                'predictions': prediction,
            }
            
            if return_targets:
                result['true_value'] = test_targets_np
            
            if state_downsample > 0:
                with timer("State extraction"):
                    result['reservoir_states'] = model.res_states.detach().cpu().numpy()[::state_downsample]
            
            results.append(result)
            print(f"RMSE: {rmse:.4f} | Iter time: {time()-iter_start:.2f}s")
            
        except Exception as e:
            print(f"Failed on combination {i}: {str(e)}")
            continue
            
        finally:
            # Cleanup
            del model
            torch.cuda.empty_cache() if 'cuda' in device else None
    
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



