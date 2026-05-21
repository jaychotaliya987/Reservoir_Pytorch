from time import time
from contextlib import contextmanager
from typing import Union, List, Tuple, Any, Optional, Dict

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

#-------------------- Suppress UserWarning --------------------------#
import warnings

warnings.filterwarnings(
    "ignore",
    message="The following arguments have no effect for a chosen solver: `jac`.",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="The following arguments have no effect for a chosen solver",
    category=UserWarning
)
#______________________________________________________________________#

def normalize_data(data):
    """
    Normalize data while preserving system dynamics,
    and centers the data in the range [-1,1]
    """
    return (data - data.min()) / (data.max() - data.min()) * 2 - 1


def discretization_with_dt(data, length, discretization=None):
    """Generate trajectory with custom time discretization when resampling == false."""
    model = data()
    if discretization is not None:
        print("discretizing manually")
        model.dt = discretization
        solution = model.make_trajectory(length, resampling=False, method="RK45")
    return solution  # type: ignore


def discretization(
    system: type,
    points_per_period_values: np.ndarray,
    trajectory_length: int,
    return_times: bool = False
) -> np.ndarray:
    """Discretize a dynamical system and return results in a structured NumPy array."""
    if not hasattr(system, 'make_trajectory'):
        raise AttributeError("System must have a 'make_trajectory' method.")
    if np.any(points_per_period_values <= 0):
        raise ValueError("All points_per_period_values must be positive.")

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
            results[i] = (pp, None)

    return results


def split(dataset: np.ndarray, window: int = 1, **kwargs):
    """
    Splits dataset into training and testing sequences offsetting
    inputs and targets with a window.
    """
    inputs, targets = dataset[:-window], dataset[window:]
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(
        inputs, targets, shuffle=False, **kwargs
    )
    train_inputs  = torch.tensor(train_inputs)
    test_inputs   = torch.tensor(test_inputs)
    train_targets = torch.tensor(train_targets)
    test_targets  = torch.tensor(test_targets)
    return train_inputs, test_inputs, train_targets, test_targets


def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error between true and predicted values."""
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _power_iteration_spectral_radius(
    W: torch.Tensor, 
    num_iterations: int = 20
) -> torch.Tensor:
    """
    Estimate max eigenvalue (spectral radius) using power iteration.
    Much faster than torch.linalg.eigvals for large matrices on RTX 4060.
    
    Args:
        W: (N, R, R) batch of matrices
        num_iterations: Number of power iterations (default 20)
    
    Returns:
        (N,) tensor of estimated spectral radii
    """
    N, R, _ = W.shape
    device = W.device
    dtype = W.dtype
    
    # Random starting vector for each matrix
    v = torch.randn(N, R, 1, device=device, dtype=dtype)
    v = v / (torch.linalg.norm(v, dim=1, keepdim=True) + 1e-9)
    
    for _ in range(num_iterations):
        # v_new = W @ v
        v_new = torch.bmm(W, v)  # (N, R, 1)
        
        # Normalize
        norm = torch.linalg.norm(v_new, dim=1, keepdim=True) + 1e-9
        v = v_new / norm
    
    # Rayleigh quotient: λ ≈ v^T @ W @ v / (v^T @ v)
    Wv = torch.bmm(W, v)  # (N, R, 1)
    vt_Wv = torch.bmm(v.transpose(1, 2), Wv)  # (N, 1, 1)
    spectral_radius = vt_Wv.squeeze(-1).abs().squeeze(-1)  # (N,)
    
    return spectral_radius


def parameter_sweep(inputs, parameter_dict,
                    return_targets=True,
                    state_downsample=-1,
                    batch_size=32,
                    **kwargs):
    """
    Batched parameter sweep using Reservoir in batched mode.

    Weight generation (random init + eigval scaling) happens ONCE before
    the loop via build_all_weights(). Each batch loop iteration just slices
    the prebuilt tensors and runs train_readout + predict — no eigval
    computation per batch.

    Args:
        inputs         : (T, input_dim) numpy array
        parameter_dict : {"SpectralRadius": array, "LeakyRate": array, "InputScaling": array}
        return_targets : include test targets in results
        state_downsample: downsample reservoir states (-1 = skip)
        batch_size     : configs per GPU batch (tune to fill VRAM)
        **kwargs       : passed to Reservoir (reservoir_dim, input_dim, output_dim, sparsity, ...)
    """

    from time import time
    import gc

    # --- 1. Data Preparation (once) ---
    with timer("Data preparation"):
        train_inputs, test_inputs, train_targets, test_targets = split(inputs, random_state=42)
        test_targets_np = test_targets.numpy() if return_targets else None
        steps_to_predict = len(test_targets)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move data to GPU once
    train_inputs  = train_inputs.to(device, non_blocking=True)
    test_inputs   = test_inputs.to(device, non_blocking=True)
    train_targets = train_targets.to(device, non_blocking=True)

    # --- 2. Extract parameter arrays ---
    sr_all  = parameter_dict["SpectralRadius"]   # (N,) numpy array
    lr_all  = parameter_dict["LeakyRate"]        # (N,) numpy array
    ins_all = parameter_dict["InputScaling"]     # (N,) numpy array
    total_combinations = len(sr_all)

    print(f"Total parameter combinations: {total_combinations}")
    print(f"Device: {device}\n")

    # Gather architectural settings from kwargs
    input_dim     = kwargs.get('input_dim',     inputs.shape[1])
    reservoir_dim = kwargs.get('reservoir_dim', 1000)
    sparsity      = kwargs.get('sparsity',      0.9)
    Iteration_approximation = kwargs.get('use_power_iteration', False)

    results = []

    # --- 3. Batch loop — weights built and evaluated per batch ---
    for batch_start in range(0, total_combinations, batch_size):
        batch_end         = min(batch_start + batch_size, total_combinations)
        actual_batch_size = batch_end - batch_start

        print(f"\n{'='*60}")
        print(f"Batch {batch_start // batch_size + 1} | "
              f"Configs {batch_start + 1}-{batch_end}/{total_combinations}")
        print(f"{'='*60}")

        batch_iter_start = time()

        try:
            # Slice hyperparameter arrays for this specific batch
            lr_batch  = lr_all[batch_start:batch_end]    
            sr_batch  = sr_all[batch_start:batch_end]
            ins_batch = ins_all[batch_start:batch_end]

            # NEW: Build ONLY the weights needed for this specific batch block
            with timer(f"Building batch weights ({actual_batch_size} configs)"):
                batch_weights = build_all_weights(
                    N                  = actual_batch_size, 
                    input_dim          = input_dim,
                    reservoir_dim      = reservoir_dim,
                    spectral_radii     = sr_batch,          # Sliced arrays passed here
                    input_scalings     = ins_batch,         # Sliced arrays passed here
                    sparsity           = sparsity,
                    device             = device,
                    use_power_iteration = Iteration_approximation
                )

            # Init model — skips full weight setup because prebuilt_weights are provided
            with timer(f"Batch init ({actual_batch_size} configs)"):
                batch_model = Reservoir(
                    spectral_radius  = sr_batch,
                    leak_rate        = lr_batch,
                    input_scaling    = ins_batch,
                    device           = device,
                    prebuilt_weights = batch_weights,
                    **{k: v for k, v in kwargs.items() if k != 'device'}
                )

            # Train readout (ridge regression) for all B configs at once
            with timer(f"Batch training ({actual_batch_size} configs)"):
                batch_model.train_readout(
                    train_inputs,
                    train_targets,
                    warmup=int(len(train_inputs) * 0.2),
                    alpha=1e-5
                )

            # Predict all B configs at once
            with timer(f"Batch prediction ({actual_batch_size} configs)"):
                with torch.no_grad():
                    batch_predictions = batch_model.predict(
                        train_inputs,
                        steps=steps_to_predict
                    )   # (steps, B, O)

            # Extract per-config results
            with timer(f"Result extraction ({actual_batch_size} configs)"):
                for config_idx in range(actual_batch_size):
                    param_idx = batch_start + config_idx
                    result = {
                        'parameters': {
                            'SpectralRadius': sr_all[param_idx],
                            'LeakyRate':      lr_all[param_idx],
                            'InputScaling':   ins_all[param_idx],
                        },
                        'predictions':    batch_predictions[:, config_idx, :].detach().cpu().numpy(),
                        'readout_weights': batch_model.W_out[config_idx].detach().cpu().numpy(),
                    }

                    if return_targets:
                        result['true_value'] = test_targets_np

                    if state_downsample > 0:
                        result['reservoir_states'] = (
                            batch_model.reservoir_states[:, config_idx, :]
                            .detach().cpu().numpy()[::state_downsample]
                        )

                    results.append(result)

            batch_iter_end = time()
            print(f"Batch time: {batch_iter_end - batch_iter_start:.2f}s "
                  f"({actual_batch_size} configs)")

        except Exception as e:
            import traceback
            print(f"Error in batch {batch_start // batch_size + 1}: {str(e)}")
            traceback.print_exc()
            raise   

        finally:
            # Aggressive, deterministic isolation and cleanup of local loop allocations
            if 'batch_model' in locals():
                del batch_model  # type: ignore
            if 'batch_weights' in locals():
                del batch_weights # type: ignore

    print(f"\n{'='*60}")
    print(f"Total combinations processed: {len(results)}/{total_combinations}")
    print(f"{'='*60}\n")

    return results


def truncate(system):
    """
    Truncate the system to the length of the least period sample.
    """
    PP_array = system['pp']
    l_period = len(system['trajectory'][-1]) // PP_array[-1]

    for i in range(len(PP_array)):
        num_points = int(l_period * PP_array[i])
        system['trajectory'][i] = system['trajectory'][i][:num_points]

    return system


def build_all_weights(
    N: int,
    input_dim: int,
    reservoir_dim: int,
    spectral_radii: np.ndarray,
    input_scalings: np.ndarray,
    sparsity: float = 0.9,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    use_power_iteration: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Build ALL reservoir weight matrices for N configs in a single pass.
    Intended to be called ONCE before a parameter sweep loop.

    The expensive operations (random generation, eigval computation,
    spectral radius scaling) happen here once instead of once per batch.

    Args:
        N              : total number of configs (e.g. 1000)
        input_dim      : input dimension
        reservoir_dim  : reservoir dimension
        spectral_radii : (N,) array of target spectral radii
        input_scalings : (N,) array of input scaling values
        sparsity       : connection sparsity (shared across all configs)
        device         : target device
        dtype          : target dtype
        use_power_iteration : If True (default), use fast power iteration to estimate 
                            spectral radius (~10x faster on RTX 4060). If False, use 
                            exact eigenvalues via torch.linalg.eigvals (slower but exact).

    Returns:
        dict with keys:
            "W_in" : (N, reservoir_dim, input_dim)  — scaled input weights
            "W"    : (N, reservoir_dim, reservoir_dim) — scaled reservoir weights
    """
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sr_t = torch.tensor(spectral_radii, device=_device, dtype=dtype)   # (N,)
    is_t = torch.tensor(input_scalings, device=_device, dtype=dtype)   # (N,)


    # --- W_in (N, R, I) ---
    W_in = (torch.rand(N, reservoir_dim, input_dim, device=_device, dtype=dtype) * 2 - 1)
    W_in = W_in * is_t[:, None, None]   # scale each config by its own input_scaling


    # --- W (N, R, R) ---
    W = torch.rand(N, reservoir_dim, reservoir_dim, device=_device, dtype=dtype) * 2 - 1
    mask = (torch.rand_like(W) > sparsity).to(dtype)
    W = W * mask


    # --- Scale by spectral radius ---
    if use_power_iteration:
        # FAST: Power iteration (Rayleigh quotient)
        # ~10x faster than exact eigenvalues.
        current_sr = _power_iteration_spectral_radius(W, num_iterations=20)
    else:
        # SLOW: Exact eigenvalues
        try:
            eigs = torch.linalg.eigvals(W)                         # (N, R) complex
            current_sr = torch.max(eigs.abs(), dim=-1).values      # (N,)  real
        except torch.linalg.LinAlgError:
            print("Warning: Eigenvalue computation failed. Using power iteration.")
            current_sr = _power_iteration_spectral_radius(W, num_iterations=20)

    current_sr = current_sr.clamp(min=1e-9)
    scale = sr_t / current_sr                              # (N,)
    W = W * scale[:, None, None]


    return {"W_in": W_in, "W": W}

#------------------ Deprecated and Testing ---------------------#


def parameter_sweep_serial(inputs, parameter_dict,
                            return_targets=True,
                            state_downsample=-1,
                            **kwargs):
    """
    Serial parameter sweep (deprecated — use parameter_sweep for GPU acceleration).
    Kept for reference and regression testing.
    """
    with timer("Data preparation"):
        train_inputs, test_inputs, train_targets, test_targets = split(inputs, random_state=42)
        test_targets_np = test_targets.numpy() if return_targets else None
        steps_to_predict = len(test_targets)

    keys_order = ["SpectralRadius", "LeakyRate", "InputScaling"]
    values = [parameter_dict[k] for k in keys_order]
    param_combinations = list(zip(*values))
    total_combinations = len(param_combinations)

    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_inputs  = train_inputs.to(device, non_blocking=True)
    test_inputs   = test_inputs.to(device, non_blocking=True)
    train_targets = train_targets.to(device, non_blocking=True)

    for i, (sr, lr, ins) in enumerate(param_combinations, 1):
        iter_start = time()
        print(f"\nCombination {i}/{total_combinations} - SR: {sr:.4f}, LR: {lr:.4f}, IS: {ins:.4f}")
        try:
            with timer("Model init"):
                model = Reservoir(
                    spectral_radius=sr,
                    leak_rate=lr,
                    input_scaling=ins,
                    **{k: v for k, v in kwargs.items() if k != 'device'}
                )

            with timer("Training"):
                model.train_readout(
                    train_inputs,
                    train_targets,
                    warmup=int(len(train_inputs) * 0.2),
                    alpha=1e-5
                )

            with timer("Prediction"):
                with torch.no_grad():
                    prediction = model.predict(
                        train_inputs,
                        steps=steps_to_predict
                    ).cpu()

            result = {
                'parameters': {'SpectralRadius': sr, 'LeakyRate': lr, 'InputScaling': ins},
                'predictions': prediction,
                'readout_weights': model.readout.weight.detach().cpu().numpy()
            }

            if return_targets:
                result['true_value'] = test_targets_np

            if state_downsample > 0:
                with timer("State extraction"):
                    result['reservoir_states'] = (
                        model.reservoir_states.detach().cpu().numpy()[::state_downsample]
                    )

            results.append(result)
            combination_time = time() - iter_start
            print(f"Time for combination {i}: {combination_time:.2f}s")

        except Exception as e:
            print(f"Failed on combination {i}: {str(e)}")
            continue

        finally:
            if 'model' in locals():
                del model  # type: ignore

    return results
