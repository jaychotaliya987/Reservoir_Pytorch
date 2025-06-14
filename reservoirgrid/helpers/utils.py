from timeit import default_timer as timer

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import Union, List, Tuple, Any
from itertools import product

import torch
from sklearn.model_selection import train_test_split
import numpy as np

from dysts.flows import *

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    splits the data for RMSE and have a option to return the test sequance and predictions for furthur use.

    Args:
        inputs: This is a plain input sequance that of type numpy.ndarray
        parameter_dict : This is a dictionary of parameters to sweep through. This only accepts 3 main parameter of the RC
                        1. Spectral Radius, 2.Leaky Rate, 3. Input Scaling in that order.
        **kwargs : This are all the parameters passed to the model.Reservoir class for generation. Intrinsically need all the parameters 
                    needed for the generation.
    returns: 
        results: A dictionary of the parameters with the prediction. Optionlly with the test sequance.

    """   

    #Parameter Combination for the loop
    param_combi = product(
        parameter_dict["SpectralRadius"],
        parameter_dict["LeakyRate"],
        parameter_dict["InputScaling"]
    )

    #spliting and generating test and train series
    train_inputs, test_inputs, train_targets, test_targets = split(inputs, random_state=42)

    results = []
    # Looping through the parameters of the dict
    for sr, lr, ins in param_combi:
        # Generate model for the parameters
        model = Reservoir(spectral_radius=sr,
                        leak_rate=lr,
                        input_scaling=ins,
                        **kwargs)

        model.train_readout(train_inputs,train_targets, warmup=int(len(train_inputs)*0.2))
        reservoir_states = (model.res_states).detach().cpu().numpy()
        with torch.no_grad():
            prediction = model.predict(train_inputs, steps=len(test_targets))

        rmse = RMSE(test_targets, prediction)

        print(f"rmse:{rmse} with params: {sr,lr,ins}")
        # Store results
        results.append({
            'parameters': {
                'SpectralRadius': sr,
                'LeakyRate': lr,
                'InputScaling': ins
            },
            'predictions': prediction.detach().cpu().numpy(),
            'true value' : test_targets.detach().cpu().numpy() if return_targets else None,
            'reservoir states': reservoir_states[::state_downsample],
            'metrics': {
                'RMSE': rmse,
            }
        })
    return results


def truncate(system):
    """
    trancate the system to the length of least period sample of the system. 

    Args:
        system: Accepts an ensamble of the same system but with different point per period.
    returns the system with the equal number of period instead of points.
    """
    PP_array = system['pp'] # Return system's points per period in a list

    l_period = len(system['trajectory'][-1])//PP_array[-1] # Length of the fewest period dataset for referance to make all sample of this exact period

    for i in range(len(PP_array)):
        num_points = int(l_period * PP_array[i]) # Calculates the points required for l_periods
        system['trajectory'][i]= system['trajectory'][i][:num_points] # Slices till the points required reached
    return system