'''
Main Reservoir class for the ReservoirGrid project.
This class implements a reservoir computing model with the following features:
- Reservoir state update with leaky integration
- Readout layer for prediction
- Training of the readout layer with ridge regression
- Optional training of the reservoir with backpropagation
- Prediction with optional teacher forcing
- Saving and loading of the model
- Echo state property checks
- Reservoir state and weights visualization and control
- Device management (CPU/GPU)

Batched mode:
    Pass 1D array/tensor for spectral_radius, leak_rate, input_scaling
    to run B reservoir configs in parallel (e.g. for hyperparameter sweeps).
    Single config (scalar args) behaviour is unchanged.

Prebuilt weights mode (sweep optimisation):
    Pass prebuilt_weights={"W_in": tensor, "W": tensor} to skip random
    generation and eigval computation entirely. Used by parameter_sweep
    to build all N reservoir matrices once before the batch loop.
'''

import torch
from torch import nn
from torch import optim

from typing import Optional, Callable, Type, Union, Dict

import numpy as np

# Default device (can be overridden)
_DEFAULT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_DEFAULT_DTYPE = torch.float32
class Reservoir(nn.Module):
    """
    Initialize the Reservoir class.
    Args:
        :param input_dim: Input dimensions. Ex: Lorenz has input_dim = 3
        :param reservoir_dim: Number of reservoir neurons.
        :param output_dim: Output dimensions. Ex: Lorenz has output_dim = 3
        :param spectral_radius: Scalar for single config, 1D array/tensor for batched mode.
        :param leak_rate: Scalar for single config, 1D array/tensor for batched mode.
        :param sparsity: 0-1, Controls the number of connections between reservoir neurons.
        :param input_scaling: Scalar for single config, 1D array/tensor for batched mode.
        :param noise_level: Noise level for the reservoir state update.
        :param activation: Activation function, defaults to tanh.
        :param device: Device to run the model on.
        :param dtype: Data type, defaults to float32.
        :param prebuilt_weights: Optional dict {"W_in": tensor, "W": tensor}.
                                 If provided, skips all weight generation and eigval
                                 computation. Tensors must already be on the correct
                                 device and dtype. Used by parameter_sweep.
    """
    def __init__(self,
                 input_dim: int,
                 reservoir_dim: int,
                 output_dim: int,
                 spectral_radius: Union[float, np.ndarray, torch.Tensor] = 0.9,
                 leak_rate: Union[float, np.ndarray, torch.Tensor] = 0.3,
                 sparsity: float = 0.9,
                 input_scaling: Union[float, np.ndarray, torch.Tensor] = 1.0,
                 noise_level: float = 0.01,
                 activation: Callable = torch.tanh,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: torch.dtype = _DEFAULT_DTYPE,
                 prebuilt_weights: Optional[Dict[str, torch.Tensor]] = None):
        super(Reservoir, self).__init__()

        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.noise_level = noise_level
        self.activation = activation
        self.device = torch.device(device) if device else _DEFAULT_DEVICE
        self.dtype = dtype

        # --- Detect batched vs single mode ---
        def _to_tensor(val):
            if isinstance(val, torch.Tensor):
                return val.to(self.device, self.dtype)
            if isinstance(val, np.ndarray):
                return torch.tensor(val, device=self.device, dtype=self.dtype)
            return torch.tensor([val], device=self.device, dtype=self.dtype)

        sr_t = _to_tensor(spectral_radius)
        lr_t = _to_tensor(leak_rate)
        is_t = _to_tensor(input_scaling)

        self.batched = sr_t.numel() > 1
        self.B       = sr_t.numel()

        self.register_buffer("spectral_radii", sr_t)
        self.register_buffer("leak_rates",     lr_t)
        self.register_buffer("input_scalings", is_t)

        # --- Parameter Validation ---
        assert torch.all(lr_t >= 0) and torch.all(lr_t <= 1), "Leak rate must be in [0, 1]"
        assert 0.0 <= sparsity <= 1.0,                         "Sparsity must be in [0, 1]"
        assert torch.all(sr_t >= 0),                           "Spectral radius must be non-negative"
        assert reservoir_dim > 0,                              "Reservoir dimension must be positive"

        # --- Initialize Weights ---
        if prebuilt_weights is not None:
            # Fast path: weights already built externally, just register them.
            # No random generation, no eigval computation.
            W_in = prebuilt_weights["W_in"].to(self.device, self.dtype)
            W    = prebuilt_weights["W"].to(self.device, self.dtype)
            assert W_in.shape == (self.B, reservoir_dim, input_dim), \
                f"prebuilt W_in shape mismatch: expected {(self.B, reservoir_dim, input_dim)}, got {W_in.shape}"
            assert W.shape == (self.B, reservoir_dim, reservoir_dim), \
                f"prebuilt W shape mismatch: expected {(self.B, reservoir_dim, reservoir_dim)}, got {W.shape}"
        else:
            # Normal path: generate weights from scratch.
            W_in = (torch.rand(self.B, reservoir_dim, input_dim, device=self.device, dtype=dtype) * 2 - 1)
            W_in = W_in * is_t[:, None, None]

            W = torch.rand(self.B, reservoir_dim, reservoir_dim, device=self.device, dtype=dtype) * 2 - 1
            mask = (torch.rand_like(W) > sparsity).to(dtype)
            W = W * mask

            try:
                eigs = torch.linalg.eigvals(W)
                current_sr = torch.max(eigs.abs(), dim=-1).values
                current_sr = current_sr.clamp(min=1e-9)
                scale = sr_t / current_sr
                W = W * scale[:, None, None]
            except torch.linalg.LinAlgError:
                print("Warning: Eigenvalue computation failed. Using unscaled reservoir weights.")

        self.W_in = nn.Parameter(W_in, requires_grad=False)   # (B, R, I)
        self.W    = nn.Parameter(W,    requires_grad=False)   # (B, R, R)

        # --- Readout layer ---
        if not self.batched:
            self.readout = nn.Linear(reservoir_dim, output_dim, device=self.device, dtype=dtype)
        else:
            self.register_buffer(
                "W_out", torch.zeros(self.B, output_dim, reservoir_dim, device=self.device, dtype=dtype)
            )
            self.register_buffer(
                "b_out", torch.zeros(self.B, output_dim, device=self.device, dtype=dtype)
            )

        # --- Initial reservoir state (B, R) ---
        self.register_buffer(
            'reservoir_states_buf',
            torch.zeros(self.B, reservoir_dim, device=self.device, dtype=dtype)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _readout(self, states: torch.Tensor) -> torch.Tensor:
        if not self.batched:
            return self.readout(states)
        return torch.einsum("tbr,bor->tbo", states, self.W_out) + self.b_out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, u: torch.Tensor, reset_state: bool = True) -> torch.Tensor:
        """
        Forward pass through the reservoir and readout layer.

        Args:
            u (torch.Tensor): Input sequence.
                Single mode:  (T, I)  or (T, 1, I)
                Batched mode: (T, I)  — same input broadcast to all B configs
            reset_state (bool): Reset reservoir state before processing.

        Returns:
            torch.Tensor:
                Single mode:  (T, O)
                Batched mode: (T, B, O)
        """
        u = u.to(self.device, self.dtype)

        if u.ndim == 3:
            u = u.squeeze(1)
        T = u.shape[0]

        if reset_state:
            self.reservoir_states_buf = torch.zeros(
                self.B, self.reservoir_dim, device=self.device, dtype=self.dtype
            )

        # Preallocate (T, B, R) — no list, no stack
        self.reservoir_states = torch.empty(
            T, self.B, self.reservoir_dim, device=self.device, dtype=self.dtype
        )

        # Preallocate full noise tensor — one kernel launch
        noise_all = torch.randn(
            T, self.B, self.reservoir_dim, device=self.device, dtype=self.dtype
        ) * self.noise_level

        leak = self.leak_rates[:, None]   # (B, 1)

        for t in range(T):
            ut = u[t]   # (I,)
            input_term     = torch.einsum("bri,i->br", self.W_in, ut)
            recurrent_term = torch.bmm(self.W, self.reservoir_states_buf.unsqueeze(-1)).squeeze(-1)
            activated = self.activation(input_term + recurrent_term + noise_all[t])
            self.reservoir_states_buf = (1.0 - leak) * self.reservoir_states_buf + leak * activated
            self.reservoir_states[t] = self.reservoir_states_buf

        output = self._readout(self.reservoir_states)

        if not self.batched:
            output = output.squeeze(1)

        return output

    # ------------------------------------------------------------------
    # Train readout
    # -----------------------------------------------------------------
    def train_readout(self,
                      inputs: torch.Tensor,
                      targets: torch.Tensor,
                      warmup: int = 0,
                      alpha: float = 1e-6,
                      chunk_size: int = 5000):
        """
        Train the readout layer using Chunked (Streaming) Ridge Regression.
        Prevents VRAM OOM errors on very long sequences by accumulating 
        the correlation matrices incrementally.
        
        Args:
            inputs: (T, I) or (T, 1, I)
            targets: (T, O)
            warmup: Number of initial timesteps to discard.
            alpha: Tikhonov regularization factor.
            chunk_size: Number of timesteps processed per GPU kernel launch.
        """
        inputs  = inputs.to(self.device, self.dtype)
        targets = targets.to(self.device, self.dtype)

        if inputs.ndim == 3:
            inputs = inputs.squeeze(1)
        
        T = inputs.shape[0]
        
        # 1. Preallocate running accumulators directly on the GPU
        # XtX shape: (B, R, R) | XtY shape: (B, R, O)
        XtX_total = torch.zeros(self.B, self.reservoir_dim, self.reservoir_dim, 
                                device=self.device, dtype=self.dtype)
        XtY_total = torch.zeros(self.B, self.reservoir_dim, self.output_dim, 
                                device=self.device, dtype=self.dtype)
        
        reset_state = True  # Only reset state on the very first chunk

        # 2. Stream through the time dimension in chunks
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            
            # Fast path: If the entire chunk falls within the warmup period,
            # just advance the reservoir state without calculating correlations.
            if end <= warmup:
                with torch.no_grad():
                    self.forward(inputs[start:end], reset_state=reset_state)
                reset_state = False
                continue
            
            # Compute reservoir states for this chunk
            with torch.no_grad():
                self.forward(inputs[start:end], reset_state=reset_state)
            reset_state = False  # Keep the state alive for subsequent chunks
            
            # Calculate how much of this specific chunk is valid past the warmup threshold
            chunk_warmup = max(0, warmup - start)
            
            X_chunk = self.reservoir_states[chunk_warmup:]  # (T_chunk_eff, B, R)
            Y_chunk = targets[start + chunk_warmup:end]     # (T_chunk_eff, O)
            
            T_chunk_eff = X_chunk.shape[0]
            if T_chunk_eff == 0:
                continue
                
            # Expand and permute to align configurations for batch matrix multiplication
            Y_chunk_exp = Y_chunk.unsqueeze(1).expand(T_chunk_eff, self.B, self.output_dim)
            X_b = X_chunk.permute(1, 0, 2)       # (B, T_chunk_eff, R)
            Y_b = Y_chunk_exp.permute(1, 0, 2)   # (B, T_chunk_eff, O)
            
            # 3. Incrementally accumulate into the total correlation buffers
            # This is mathematically identical to running it all at once!
            # TO THESE:
            XtX_total.add_(torch.bmm(X_b.transpose(1, 2), X_b))
            XtY_total.add_(torch.bmm(X_b.transpose(1, 2), Y_b))
            

        # 4. Apply Tikhonov regularization and solve the accumulated global system
        I = torch.eye(self.reservoir_dim, device=self.device, dtype=self.dtype).unsqueeze(0)
        A = XtX_total + alpha * I

        try:
            solution = torch.linalg.solve(A, XtY_total)   # (B, R, O)
        except torch.linalg.LinAlgError:
            print("Warning: solve failed, falling back to lstsq.")
            solution = torch.linalg.lstsq(A, XtY_total).solution

        # 5. Overwrite readout layer parameters
        with torch.no_grad():
            if not self.batched:
                self.readout.weight.copy_(solution.squeeze(0).T)
                if self.readout.bias is not None:
                    self.readout.bias.zero_()
            else:
                self.W_out.copy_(solution.transpose(1, 2))
                self.b_out.zero_()

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self,
                initial_input: torch.Tensor,
                steps: int,
                teacher_forcing_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Autonomous prediction after warming up on initial_input.
        """
        initial_input = initial_input.to(self.device, self.dtype)
        if initial_input.ndim == 3:
            initial_input = initial_input.squeeze(1)

        if self.output_dim != self.input_dim and teacher_forcing_targets is None:
            raise ValueError("output_dim must match input_dim for autonomous prediction.")

        self.eval()

        with torch.no_grad():
            self.forward(initial_input, reset_state=True)

            if not self.batched:
                current = self.readout(self.reservoir_states_buf)
            else:
                current = (torch.einsum("br,bor->bo", self.reservoir_states_buf, self.W_out)
                           + self.b_out)

            leak = self.leak_rates[:, None]
            predictions = []

            for step in range(steps):
                input_term     = torch.einsum("bri,bi->br", self.W_in, current)
                recurrent_term = torch.bmm(self.W, self.reservoir_states_buf.unsqueeze(-1)).squeeze(-1)
                activated      = self.activation(input_term + recurrent_term)
                self.reservoir_states_buf = (1.0 - leak) * self.reservoir_states_buf + leak * activated

                if not self.batched:
                    pred = self.readout(self.reservoir_states_buf)
                else:
                    pred = (torch.einsum("br,bor->bo", self.reservoir_states_buf, self.W_out)
                            + self.b_out)

                predictions.append(pred)

                if teacher_forcing_targets is not None and step < teacher_forcing_targets.size(0):
                    current = teacher_forcing_targets[step].to(self.device, self.dtype)
                    if current.ndim == 1:
                        current = current.unsqueeze(0).expand(self.B, -1)
                else:
                    current = pred

        result = torch.stack(predictions, dim=0)

        if not self.batched:
            result = result.squeeze(1)

        return result

    # ------------------------------------------------------------------
    # Reservoir control
    # ------------------------------------------------------------------

    def update_reservoir(self, u: torch.Tensor):
        print("Warning: `update_reservoir` will update the reservoir states manually.")
        self.reservoir_states_buf = u
        self.reservoir_states = torch.cat(
            (self.reservoir_states, self.reservoir_states_buf.unsqueeze(0)), dim=0
        )

    def freeze_reservoir(self):
        self.W_in.requires_grad = False
        self.W.requires_grad    = False
        print("Reservoir weights (W_in, W) frozen.")

    def unfreeze_reservoir(self):
        self.W_in.requires_grad = True
        self.W.requires_grad    = True
        print("Reservoir weights (W_in, W) unfrozen.")

    def reset_state(self, batch_size: int = 1):
        b = batch_size if batch_size > 0 else self.B
        self.reservoir_states_buf = torch.zeros(
            b, self.reservoir_dim, device=self.device, dtype=self.dtype
        )
        print(f"Reservoir state reset for batch size {b}.")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str, map_location: Optional[Union[str, torch.device]] = None):
        if map_location is None:
            map_location = self.device
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.to(self.device)
        print(f"Model loaded from {path} to device {self.device}")

    # ------------------------------------------------------------------
    # Finetune — single mode only
    # ------------------------------------------------------------------

    def finetune(self,
                 inputs: torch.Tensor,
                 targets: torch.Tensor,
                 epochs: int,
                 lr: float,
                 criterion_class: Type[nn.Module] = nn.MSELoss,
                 optimizer_class: Type[optim.Optimizer] = optim.Adam,
                 print_every: int = 10):
        """
        Fine-tunes the entire model via backpropagation.
        Single mode only — batched mode uses train_readout for analytical fitting.
        TODO: IMPLEMENTATION
        """

    # ------------------------------------------------------------------
    # In-Class Hyperparameter Optimization Routine
    # ------------------------------------------------------------------

    def optimize(self,
                 X_train: torch.Tensor,
                 Y_train: torch.Tensor,
                 X_val: torch.Tensor,
                 Y_val: torch.Tensor,
                 metric_fn: Callable,
                 n_trials: int = 100,
                 batch_size: int = 10,
                 direction: str = "minimize",
                 warmup: int = 100,
                 alpha: float = 1e-5) -> dict:
        """
        Optimizes the reservoir's continuous hyperparameters using an external metric function.
        After optimization, updates this specific instance with the best parameters and weights.

        Args:
            X_train, Y_train: Training sequence tensors.
            X_val, Y_val: Validation sequence tensors.
            metric_fn: Injected evaluation function from your toolkit (chaos_utils).
            n_trials: Total optimization steps.
            batch_size: Number of configurations evaluated in parallel on the GPU.
            direction: "minimize" or "maximize" depending on the metric_fn.
            warmup: Number of initial transient timesteps to discard.
            alpha: Tikhonov regularization factor for train_readout.
        """
        import optuna
        sampler = optuna.samplers.TPESampler()

        # Enforce that training/validation tensors match the model's dtype and device
        X_train = X_train.to(self.device, self.dtype)
        Y_train = Y_train.to(self.device, self.dtype)
        X_val = X_val.to(self.device, self.dtype)
        Y_val = Y_val.to(self.device, self.dtype)

        study = optuna.create_study(direction=direction, sampler=sampler)
        num_batches = int(np.ceil(n_trials / batch_size))

        print(f"[{type(self).__name__}] Starting in-class optimization using '{metric_fn.__name__}' ({direction} mode)...")

        for b_idx in range(num_batches):
            trials = [study.ask() for _ in range(min(batch_size, n_trials - len(study.trials)))]
            if not trials:
                break

            # 1. Collect hyperparameter sweeps from Optuna
            sr_list, lr_list, is_list = [], [], []
            for trial in trials:
                sr_list.append(trial.suggest_float("spectral_radius", 0.1, 1.5, step=0.01))
                lr_list.append(trial.suggest_float("leak_rate", 0.05, 1.0, step=0.05))
                is_list.append(trial.suggest_float("input_scaling", 0.1, 1.0, step=0.01))

            # 2. Instantiate a temporary batched Reservoir using this instance as a blueprint
            # Topologically similar to sweep
            search_batch = Reservoir(
                input_dim=self.input_dim,
                reservoir_dim=self.reservoir_dim,
                output_dim=self.output_dim,
                spectral_radius=np.array(sr_list),
                leak_rate=np.array(lr_list),
                input_scaling=np.array(is_list),
                sparsity=self.sparsity,
                noise_level=self.noise_level,
                activation=self.activation,
                device=self.device,
                dtype=self.dtype
            )

            # 3. Fit readout systems simultaneously across the batch
            search_batch.train_readout(inputs=X_train, targets=Y_train, warmup=warmup, alpha=alpha)

            # 4. Evaluate predictions on validation sequence [Shape: (T, B, O)]
            search_batch.eval()
            with torch.no_grad():
                val_predictions = search_batch(X_val, reset_state=True)

            # 5. Score slices using your injected custom metric function
            for i, trial in enumerate(trials):
                pred_slice = val_predictions[:, i, :]
                
                # Dynamic format catch: handle if your chaos_utils expect numpy arrays or raw tensors
                try:
                    score = metric_fn(Y_val, pred_slice)
                except TypeError:
                    # Fallback if the function natively requires numpy arrays
                    score = metric_fn(Y_val.cpu().numpy(), pred_slice.cpu().numpy())

                study.tell(trial, score)

        print(f"Optimization complete! Best Parameters: {study.best_params}")

        # --- UPDATE THE CURRENT INSTANCE STATE ---
        print(f"Re-initializing current instance weights with the optimal configuration...")
        
        # Turn off batch mode flags on this specific instance
        self.batched = False
        self.B = 1
        
        # Update your buffers to the optimal scalars found by Optuna
        self.register_buffer("spectral_radii", torch.tensor([study.best_params["spectral_radius"]], device=self.device, dtype=self.dtype))
        self.register_buffer("leak_rates", torch.tensor([study.best_params["leak_rate"]], device=self.device, dtype=self.dtype))
        self.register_buffer("input_scalings", torch.tensor([study.best_params["input_scaling"]], device=self.device, dtype=self.dtype))

        # Re-generate clean, single-mode matrices scaled to the exact winning dimensions
        W_in_opt = (torch.rand(1, self.reservoir_dim, self.input_dim, device=self.device, dtype=self.dtype) * 2 - 1)
        W_in_opt = W_in_opt * self.input_scalings[:, None, None]

        W_opt = torch.rand(1, self.reservoir_dim, self.reservoir_dim, device=self.device, dtype=self.dtype) * 2 - 1
        mask = (torch.rand_like(W_opt) > self.sparsity).to(self.dtype)
        W_opt = W_opt * mask

        eigs = torch.linalg.eigvals(W_opt)
        current_sr = torch.max(eigs.abs(), dim=-1).values.clamp(min=1e-9)
        W_opt = W_opt * (self.spectral_radii / current_sr)[:, None, None]

        # Overwrite the base parameter weights
        self.W_in = nn.Parameter(W_in_opt, requires_grad=False)
        self.W = nn.Parameter(W_opt, requires_grad=False)

        # Build a single-mode linear readout layer instead of the batched tensor buffer
        self.readout = nn.Linear(self.reservoir_dim, self.output_dim, device=self.device, dtype=self.dtype)
        
        # Train this optimized single configuration one final time so the model is ready to predict immediately
        self.train_readout(inputs=X_train, targets=Y_train, warmup=warmup, alpha=alpha)
        
        return study.best_params


