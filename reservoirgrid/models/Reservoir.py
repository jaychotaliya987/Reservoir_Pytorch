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
    # ------------------------------------------------------------------

    def train_readout(self,
                      inputs: torch.Tensor,
                      targets: torch.Tensor,
                      warmup: int = 0,
                      alpha: float = 1e-6):
        """
        Train the readout layer using Ridge Regression.
        Works in both single and batched mode.
        """
        inputs  = inputs.to(self.device, self.dtype)
        targets = targets.to(self.device, self.dtype)

        with torch.no_grad():
            self.forward(inputs, reset_state=True)
            X = self.reservoir_states[warmup:]    # (T', B, R)
            Y = targets[warmup:]                  # (T', O)

        T_eff = X.shape[0]
        Y_exp = Y.unsqueeze(1).expand(T_eff, self.B, self.output_dim)

        X_b = X.permute(1, 0, 2)       # (B, T', R)
        Y_b = Y_exp.permute(1, 0, 2)   # (B, T', O)

        XtX = torch.bmm(X_b.transpose(1, 2), X_b)   # (B, R, R)
        XtY = torch.bmm(X_b.transpose(1, 2), Y_b)   # (B, R, O)

        I = torch.eye(self.reservoir_dim, device=self.device, dtype=self.dtype).unsqueeze(0)
        A = XtX + alpha * I

        try:
            solution = torch.linalg.solve(A, XtY)   # (B, R, O)
        except torch.linalg.LinAlgError:
            print("Warning: solve failed, falling back to lstsq.")
            solution = torch.linalg.lstsq(A, XtY).solution

        with torch.no_grad():
            if not self.batched:
                self.readout.weight.copy_(solution.squeeze(0).T)
                if self.readout.bias is not None:
                    self.readout.bias.zero_()
            else:
                self.W_out = solution.transpose(1, 2)   # (B, O, R)
                self.b_out.zero_()

        print("Readout training complete.")

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
                 print_every: int = 10) -> torch.Tensor:
        """
        Fine-tunes the entire model via backpropagation.
        Single mode only — batched mode uses train_readout for analytical fitting.
        """
        if self.batched:
            raise RuntimeError("finetune() is for single-config mode only. "
                               "Use train_readout() for batched sweeps.")

        if not self.W_in.requires_grad or not self.W.requires_grad:
            print("Warning: reservoir weights are frozen. Call unfreeze_reservoir() first.")

        inputs  = inputs.to(self.device, self.dtype)
        targets = targets.to(self.device, self.dtype)

        criterion = criterion_class()
        #optimizer = optimizer_class(self.parameters(), lr=lr)
        losses    = torch.zeros(epochs, device=self.device)

        self.train()
        for epoch in range(epochs):
            #optimizer.zero_grad()
            output = self(inputs, reset_state=True)
            loss   = criterion(output, targets)
            loss.backward()
            #optimizer.step()
            losses[epoch] = loss.item()
            if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
                print(f'Finetune Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}')

        self.eval()
        return losses

    def best_hyperparameters(self):
        pass