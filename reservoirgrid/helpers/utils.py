from timeit import default_timer as timer

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import Union
from typing import List, Tuple, Any
import torch

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from dysts.flows import *

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
        print("discretizing manualy")
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
    splits dataset into training and testing sequance offsetting
    inputs and targets with a window. generally 1, but can be overwritten.
    The inputs are also converted to the torch.tensor type.
    accepts **kwargs, passed to train_test_split
    
    Args:
        dataset : input dataset, accepts dtypes accepted by sklearn
        window : offsetting parameter. Target is offsetted by window in the future
    Returns:
        train_inputs, test_inputs, train_targets, test_targets 

    """
    inputs, targets = dataset[:-window], dataset[window:]
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, shuffle=False, **kwargs)
    train_inputs = torch.tensor(train_inputs)
    test_inputs = torch.tensor(test_inputs)
    train_targets = torch.tensor(train_targets)
    test_targets = torch.tensor(test_targets)
    return train_inputs, test_inputs, train_targets, test_targets 