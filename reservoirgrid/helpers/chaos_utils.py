import torch
from sklearn.model_selection import train_test_split
from scipy.special import rel_entr
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from reservoirgrid.models import Reservoir
from reservoirgrid.datasets import MackeyGlassDataset
from reservoirgrid.helpers import utils

def test_lyapunov(Model):
    # Get reservoir state norms
    state_norms = [state.norm().item() for state in Model.res_states]
    
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

def lyapunov_time(series, threshold=0.1, min_samples=10):
    """
    Calculate the Lyapunov time from prediction errors.
    
    The Lyapunov time estimates how long we can reasonably predict the system before
    errors grow beyond an acceptable threshold.
    
    Parameters:
    -----------
    pred : torch.Tensor or np.ndarray
        Prediction values (T x output_dim)
    threshold : float, optional
        Divergence threshold (default: 0.1)
    min_samples : int, optional
        Minimum number of samples required for valid calculation (default: 10)
    
    Returns:
    --------
    float
        Lyapunov time in steps
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(series, torch.Tensor):
        series = series.detach().cpu().numpy()
    
    # Ensure we have enough samples
    if len(series) < min_samples:
        return len(series)
    
    # Calculate relative error growth
    errors = np.abs(series - series[0])  # Distance from initial state
    normalized_errors = errors / (np.abs(series[0]) + 1e-10)  # Relative error
    
    # Find first time error exceeds threshold
    for i in range(1, len(normalized_errors)):
        if np.any(normalized_errors[i] > threshold):
            return i
    
    return len(series)

def comparative_lyapunov_time(test_targets, predictions, threshold=0.1):
    """
    Computes Lyapunov time as the time when prediction diverges from ground truth.

    Args:
        test_targets: true trajectory (T, D)
        predictions: predicted trajectory (T, D)
        threshold: divergence threshold (e.g., 0.1 for normalized data)

    Returns:
        Integer time index when prediction error exceeds threshold
    """
    errors = np.linalg.norm(predictions - test_targets, axis=1)
    diverged = np.where(errors > threshold)[0]
    return diverged[0] if len(diverged) > 0 else len(errors)

def lyapunov_time_from_fit(true, pred, fit_range=0.5):
    """Calculate Lyapunov time via exponential fit of error growth"""
    errors = np.linalg.norm(pred - true, axis=1)
    t = np.arange(len(errors))
    
    # Only use first portion of data for fitting
    fit_len = int(len(errors) * fit_range)
    if fit_len < 2:
        return len(pred)
    
    # Fit log errors to linear model
    with np.errstate(divide='ignore'):
        log_err = np.log(errors[:fit_len] + 1e-10)
    slope = np.polyfit(t[:fit_len], log_err, 1)[0]
    
    # Lyapunov time is inverse of slope
    return 1.0 / slope if slope > 0 else float('inf')


def KLdivergence(true:np.ndarray, predicted: np.ndarray, bins: int):
    """
    Calculates the Kullback–Leibler (KL) divergence of the true and predicted system
    
    Arguments:
        true: Numpy array of test_targets
        predicted: Numpy array of predicted array from the reservoir
        bins: number of bins for histogram generation
    Returns: 
        KLDivergence: a float    
    """
    all_data = np.vstack([true, predicted])

    # Compute common bin edges
    ranges = [(np.min(all_data[:, i]), np.max(all_data[:, i])) for i in range(all_data.shape[1])]

    H_true, _ = np.histogramdd(true, bins=bins, range=ranges, density=True)
    H_pred, _ = np.histogramdd(predicted, bins=bins, range=ranges, density=True)

    # Flatten and normalize
    P = H_true.flatten() + 1e-10
    Q = H_pred.flatten() + 1e-10

    P /= P.sum()
    Q /= Q.sum()
    
    kl_div = np.sum(rel_entr(P, Q))
    
    return kl_div.item()


def correlation_dimension(data, r_vals):
    """
    """
    N = len(data)
    dists = squareform(pdist(data))  # pairwise distances
    C = []
    for r in r_vals:
        count = np.sum(dists < r) - N  # remove diagonal
        C_r = count / (N * (N - 1))
        C.append(C_r)
    return np.array(C)


def comparive_correlation_dim(true: np.ndarray , prediction:np.ndarray,  r_vals:np.ndarray = np.logspace(-3,0,50)):
    """
    """
    pred_c = correlation_dimension(predictions, r_vals)
    true_c = correlation_dimension(truth, r_vals)
    
    fit_range = (r_vals > 0.01) & (r_vals < 0.1)
    slope1 = np.polyfit(np.log(r_vals[fit_range]), np.log(pred_c[fit_range]), 1)[0]
    slope2 = np.polyfit(np.log(r_vals[fit_range]), np.log(true_c[fit_range]), 1)[0]

    corr = slope1 - slope2

    return corr


def psd_errors(true: np.ndarray , prediction:np.ndarray, cos_sim:bool =False):
    """
    returns Power spectrum errors and/or cosine simiilarity of the ground truth and predictions
    
    """
    f, P_true = welch(true[:, 0], fs=1.0, nperseg=1024)
    _, P_pred = welch(prediction[:, 0], fs=1.0, nperseg=1024)

    psd_error = np.linalg.norm(P_true - P_pred)
    cos_sim = cosine_similarity(P_true.reshape(1, -1), P_pred.reshape(1, -1))[0][0]
    return psd_error, cos_sim if cos_sim else psd_error

if __name__ == "__main__":
    
    Model = Reservoir(
    input_dim=1,
    reservoir_dim=1300,
    output_dim=1,
    spectral_radius=1,
    leak_rate=0.5,
    sparsity=0.9,
    input_scaling=0.5,
    noise_level = 0.01)
    
    Mglass1 = MackeyGlassDataset(10000, 5, tau=17, seed=0)
    inputs, targets = Mglass1[0]
    inputs, targets = utils.normalize_data(inputs), utils.normalize_data(targets)
    train_inputs, test_inputs = train_test_split(inputs, test_size = 0.2, shuffle=False)
    train_targets, test_targets = train_test_split(targets, test_size = 0.2, shuffle=False)

    Model.train_readout(train_inputs, train_targets, warmup=200)

    test_lyapunov(Model)

    predictions = Model.predict(train_inputs, steps = len(test_targets))
    print(lyapunov_time(predictions.cpu()))
    print(lyapunov_time_from_fit(pred=predictions.cpu()))
    plt.plot(np.log(errors + 1e-10))
    plt.title("log(Error) vs Time")
    plt.xlabel("t")
    plt.ylabel("log(error)")