import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.signal import welch
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional, Union

# Constants
EPSILON = 1e-10
MIN_SAMPLES = 10

def lyapunov_exponent(model_states: list) -> float:
    """Compute Lyapunov exponent from reservoir state norms."""
    state_norms = np.array([state.norm().item() for state in model_states])
    diffs = np.abs(np.diff(state_norms))
    
    if len(diffs) < MIN_SAMPLES:
        return 0.0
    
    lyap_exp = np.mean(np.log(diffs + EPSILON))
    return float(lyap_exp)

def lyapunov_time(
    truth: Union[np.ndarray, torch.Tensor], 
    predictions: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.1, 
    method: str = 'threshold',
    min_samples: int = 50,  # Minimum points for fitting
    max_samples: int = 1000,  # Avoid overfitting long trajectories
) -> float:
    
    if isinstance(truth, torch.Tensor):
        truth = truth.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    if method == 'threshold':
        errors = np.abs(truth - predictions)
        normalized_errors = errors / (np.abs(predictions) + EPSILON)
        exceed_idx = np.argmax(normalized_errors > threshold)
        return float(exceed_idx if exceed_idx > 0 else len(truth))

    elif method == 'fit':
        errors = np.linalg.norm(truth - predictions, axis=-1)
        
        # Ensure errors are not too small or zero
        valid_errors = errors[errors > 1e-10]  # Discard near-zero errors
        if len(valid_errors) < min_samples:
            return np.inf  # Not enough data
        
        t = np.arange(len(valid_errors))
        fit_len = min(max_samples, len(valid_errors) // 2)  # Avoid overfitting
        
        # Fit log-error vs time
        with np.errstate(divide='ignore', invalid='ignore'):
            slope, intercept = np.polyfit(t[:fit_len], np.log(valid_errors[:fit_len]), 1)
        
        # Only accept positive slopes (exponential growth)
        if slope <= 1e-10:  # Avoid division by zero or negative slopes
            return np.inf
        else:
            return float(1.0 / slope)

    else:
        raise ValueError(f"Invalid method: '{method}'. Use 'threshold' or 'fit'.")

        
def kl_divergence(
    truth: np.ndarray, 
    predictions: np.ndarray, 
    bins: int = 100
) -> float:
    """Optimized KL divergence calculation."""
    all_data = np.vstack([truth, predictions])
    ranges = [(all_data[:, i].min(), all_data[:, i].max()) for i in range(all_data.shape[1])]
    
    # Use optimized histogramdd
    H_true, _ = np.histogramdd(truth, bins=bins, range=ranges, density=True)
    H_pred, _ = np.histogramdd(predictions, bins=bins, range=ranges, density=True)
    
    P = (H_true.flatten() + EPSILON) / (H_true.sum() + EPSILON * bins**truth.shape[1])
    Q = (H_pred.flatten() + EPSILON) / (H_pred.sum() + EPSILON * bins**truth.shape[1])
    
    return np.sum(np.where(P > 0, P * np.log(P / Q), 0))

def correlation_dimension(
    data: np.ndarray, 
    r_vals: np.ndarray = np.logspace(-3, 0, 50),
    batch_size: Optional[int] = None
) -> np.ndarray:
    """Memory-efficient correlation dimension."""
    N = len(data)
    if batch_size and N > batch_size:
        # Batch processing for large data
        C = []
        for r in r_vals:
            count = 0
            for i in range(0, N, batch_size):
                batch = data[i:i+batch_size]
                dists = pdist(batch, 'euclidean')
                count += np.sum(dists < r)
            C.append(2 * count / (N * (N - 1)))
        return np.array(C)
    else:
        dists = pdist(data)
        return np.array([np.sum(dists < r) * 2 / (N * (N - 1)) for r in r_vals])

def psd_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    nperseg: int = 1024
) -> Tuple[float, float]:
    """Compute PSD error and cosine similarity in one pass."""
    f, P_true = welch(truth[:, 0], fs=1.0, nperseg=nperseg)
    _, P_pred = welch(predictions[:, 0], fs=1.0, nperseg=nperseg)
    
    psd_error = np.linalg.norm(P_true - P_pred)
    cos_sim = cosine_similarity(P_true.reshape(1, -1), P_pred.reshape(1, -1))[0, 0]
    
    return float(psd_error), cos_sim