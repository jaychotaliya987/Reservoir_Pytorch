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
'''

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Optional, Callable, Type, Union

# Default device (can be overridden)
_DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DEFAULT_DTYPE = torch.float64

class Reservoir(nn.Module):
    """
    Initialize the Reservoir class.
    Args:
        :param input_dim: Input dimensions. Ex: Lorenz has input_dim = 3
        :param reservoir_dim: Number of reservoir neurons. Keep it as big as computationally possible.
        :param output_dim: Output dimensions. Ex: Lorenz has output_dim = 3
        :param spectral_radius: Highest eigenvalue of the reservoir weight matrix
                                controls the memory of the reservoir. 
                                0-1 for stability. Higher value for more memory.
        :param leak_rate: 0-1, controls the leakiness of the reservoir. 
                          1 = no leak, Fully replaces the state 
                          0 = full leak, Slow update    
        :param sparsity: 0-1, Controls the number of connections between the reservoir neurons.
                         0 = fully connected, 1 = no connections
        :param input_scaling: Scaling factor for the input weights
                         0-1, Controls the scaling of the input weights.  
                         Higher for more input drive or When the y-variance is low
        :param noise_level: Noise level for the reservoir state update, 
                            Higher for more generalized learning
        :param activation: Activation function for the reservoir neurons, defaults to tanh.
        :param device: Device to run the model on (CPU or GPU), 
        :param dtype: Data type for the model parameters, defaults to float64.
    """
    def __init__(self,
             input_dim: int,
             reservoir_dim: int,
             output_dim: int,
             spectral_radius: float = 0.9,
             leak_rate: float = 0.3,
             sparsity: float = 0.9,
             input_scaling: float = 1.0,
             noise_level: float = 0.01,
             activation: Callable = torch.tanh,
             device: Optional[Union[str, torch.device]] = None,
             dtype: torch.dtype = _DEFAULT_DTYPE):
        super(Reservoir, self).__init__()

        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.noise_level = noise_level
        self.activation = activation
        self.device = torch.device(device) if device else _DEFAULT_DEVICE
        self.dtype = dtype

        # --- Parameter Validation ---
        assert 0.0 <= self.leak_rate <= 1.0, "Leak rate must be in [0, 1]"
        assert 0.0 <= self.sparsity <= 1.0, "Sparsity must be in [0, 1]"
        assert self.spectral_radius >= 0.0, "Spectral radius must be non-negative"
        assert self.reservoir_dim > 0, "Reservoir dimension must be positive"

        # --- Initialize Weights with scaling ---
        self.W_in = (torch.rand(reservoir_dim, input_dim, device=self.device, dtype=self.dtype) * 2 - 1) * self.input_scaling

        # Initialize sparse reservoir weights W
        W_candidate = torch.rand(reservoir_dim, reservoir_dim, device=self.device, dtype=self.dtype) * 2 - 1
        # Create sparse mask
        mask = torch.rand(reservoir_dim, reservoir_dim, device=self.device, dtype=self.dtype) > self.sparsity
        W_candidate *= mask.float() # Apply sparsity

        # Scale spectral radius using torch.linalg.eigvals
        if self.spectral_radius > 0 and self.reservoir_dim > 0:
            try:
                # Note: eigvals can return complex numbers
                eigenvalues = torch.linalg.eigvals(W_candidate)
                current_spectral_radius = torch.max(torch.abs(eigenvalues))
                # Add epsilon for numerical stability if radius is very small
                if current_spectral_radius < 1e-9:
                    print("Warning: Reservoir matrix spectral radius is close to zero.")
                    self.W = W_candidate # Use unscaled if radius is ~0
                else:
                    self.W = W_candidate * (self.spectral_radius / (current_spectral_radius + 1e-9)) # Add eps
            except torch.linalg.LinAlgError:
                 print("Warning: Eigenvalue computation failed. Using unscaled reservoir weights.")
                 self.W = W_candidate # Fallback
        else:
             self.W = W_candidate # Use unscaled if spectral_radius is 0

        # --- Make W_in and W non-trainable by default (ESN property) ---
        self.W_in = nn.Parameter(self.W_in, requires_grad=False)
        self.W = nn.Parameter(self.W, requires_grad=False)

        # --- Readout layer (trainable) ---
        self.readout = nn.Linear(reservoir_dim, output_dim, device=self.device, dtype=self.dtype)

        # --- Initialize state (as buffer on target device) ---
        self.register_buffer('reservoir_state', torch.zeros(reservoir_dim, device=self.device, dtype=self.dtype))


    def forward(self, u: torch.Tensor, reset_state: bool = True) -> torch.Tensor: #? return: torch.Tensor
        """
        Forward pass through the reservoir and readout layer.

        Args:
            u (torch.Tensor): Input sequence (SequenceLength x BatchSize x InputDim)
                               or (SequenceLength x InputDim), Then Batch size is handled internally.
            reset_state (bool): If True, reset the reservoir state before processing the sequence.

        Returns:
            torch.Tensor: Output sequence (SequenceLength x BatchSize x OutputDim or SequenceLength x OutputDim).
        """

        # --- Input Handling and Device Checks ---
        if u.device != self.device:
            print(f"Warning: inputs are not on the same device as model moving inputs to {_DEFAULT_DEVICE}")
            u = u.to(self.device)
        if u.dtype != self.dtype:
             print(f"Warning: Input tensor dtype ({u.dtype}) differs from model dtype ({self.dtype}). Casting input.")
             u = u.to(self.dtype)

        # Batch dimension - Batch Dimensions are 3 as a convention. If it is of atleast 3 it is already batched. 
        # Keep in mind to use the convention for inputs. 
        batched_input = u.ndim == 3
        if not batched_input:
            u = u.unsqueeze(1) # Add batch dimension: T x 1 x Dim #? Although no time axis, It is a good thing to have a batch dimension like this.

        batch_size = u.size(1)
        seq_len = u.size(0)

        # --- State Reset ---
        if reset_state or not hasattr(self, 'reservoir_state') or self.reservoir_state.size(0) != batch_size * self.reservoir_dim:
             # Reshape state for batch processing: (BatchSize * ReservoirDim)
             # Or maybe (BatchSize x ReservoirDim)?
             self.reservoir_state = torch.zeros(batch_size, self.reservoir_dim, device=self.device, dtype=self.dtype)

        # --- Process Sequence ---
        collected_states = []
        for t in range(seq_len):
            ut = u[t] # BatchSize x InputDim

            # Add small noise for regularization
            noise = torch.randn_like(self.reservoir_state) * self.noise_level

            # Calculate pre-activation state: (BatchSize x ReservoirDim)
            # W_in: ReservoirDim x InputDim
            # ut: BatchSize x InputDim -> ut.T: InputDim x BatchSize
            # W: ReservoirDim x ReservoirDim
            # reservoir_state: BatchSize x ReservoirDim -> reservoir_state.T: ReservoirDim x BatchSize
            # Result needs to be BatchSize x ReservoirDim
            input_term = torch.matmul(ut, self.W_in.T)  # BatchSize x ReservoirDim
            recurrent_term = torch.matmul(self.reservoir_state, self.W.T) # BatchSize x ReservoirDim

            pre_activation = input_term + recurrent_term + noise
            activated_state = self.activation(pre_activation) # BatchSize x ReservoirDimh
            # Update reservoir state with leaky integration
            self.reservoir_state = ((1 - self.leak_rate) * self.reservoir_state +
                                    self.leak_rate * activated_state)

            collected_states.append(self.reservoir_state) # Store state for this time step

        # Stack collected states: SeqLen x BatchSize x ReservoirDim
        self.res_states = torch.stack(collected_states, dim=0)

        # --- Apply Readout ---
        # Reshape states if needed for linear layer: (SeqLen * BatchSize) x ReservoirDim
        # readout expects (N, *, H_in), where H_in is reservoir_dim
        output = self.readout(self.res_states) # SeqLen x BatchSize x OutputDim

        # Remove batch dimension if input was not batched
        if not batched_input:
            output = output.squeeze(1) # T x OutputDim

        return output

    def train_readout(self,
                      inputs: torch.Tensor,
                      targets: torch.Tensor,
                      warmup: int = 0,
                      alpha: float = 1e-6):
        """
        Train the readout layer using Ridge Regression (Tikhonov regularization).

        Args:
            inputs (torch.Tensor): Input sequence (SequenceLength x [BatchSize x] InputDim).
            targets (torch.Tensor): Target sequence (SequenceLength x [BatchSize x] OutputDim).
            warmup (int): Number of initial time steps to discard before training.
            alpha (float): Ridge regularization parameter (lambda).
        """
        if inputs.device != self.device or targets.device != self.device:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        if inputs.dtype != self.dtype or targets.dtype != self.dtype:
             print(f"Warning: Input/Target dtypes ({inputs.dtype}/{targets.dtype}) differ from model dtype ({self.dtype}). Casting.")
             inputs = inputs.to(self.dtype)
             targets = targets.to(self.dtype)

        # --- Collect Reservoir States ---
        with torch.no_grad():
            # Run forward pass to populate reservoir states, reset state beforehand
            self.forward(inputs, reset_state=False)
            # Get the collected states (SeqLen x BatchSize x ReservoirDim or SeqLen x ReservoirDim)
            X = self.res_states

        # --- Handle Optional Batch Dimension ---
        batched = X.ndim == 3
        if not batched:
            X = X.unsqueeze(1) # T x 1 x R_dim
            targets = targets.unsqueeze(1) # T x 1 x O_dim

        seq_len, batch_size, _ = X.shape

        # --- Apply Warmup, Discards inputs ---
        if warmup > 0:
            if warmup >= seq_len:
                raise ValueError(f"Warmup ({warmup}) cannot be >= sequence length ({seq_len})")
            X_train = X[warmup:]
            y_train = targets[warmup:]
        else:
            X_train = X
            y_train = targets

        # Reshape for regression: (EffectiveSeqLen * BatchSize) x ReservoirDim
        X_flat = X_train.reshape(-1, self.reservoir_dim)
        # Reshape targets: (EffectiveSeqLen * BatchSize) x OutputDim
        y_flat = y_train.reshape(-1, self.output_dim)

        # --- Perform Ridge Regression using PyTorch ---
        # W_out = (X^T X + alpha * I)^(-1) X^T y
        # Solve (X^T X + alpha * I) W_out = X^T y for W_out
        XtX = torch.matmul(X_flat.T, X_flat)
        I = torch.eye(self.reservoir_dim, device=self.device, dtype=self.dtype)
        Xty = torch.matmul(X_flat.T, y_flat)

        # Use torch.linalg.solve for numerical stability and potential efficiency
        try:
            solution = torch.linalg.solve(XtX + alpha * I, Xty) # Shape: ReservoirDim x OutputDim
        except torch.linalg.LinAlgError:
             print("Warning: Linear system solving failed (matrix might be singular). "
                   "Trying pseudo-inverse.")
             # Fallback using pseudo-inverse (more robust but potentially slower)
             A = XtX + alpha * I
             A_pinv = torch.linalg.pinv(A)
             solution = torch.matmul(A_pinv, Xty)


        # --- Update Readout Weights ---
        with torch.no_grad():
            # solution contains the weights (ReservoirDim x OutputDim)
            self.readout.weight.copy_(solution.T) # Transpose to match nn.Linear: OutputDim x ReservoirDim
            # Reset bias (common practice, though bias could be solved for by augmenting X)
            if self.readout.bias is not None:
                self.readout.bias.zero_()
        print("Readout training complete.")

    def predict(self,
                initial_input: torch.Tensor,
                steps: int,
                teacher_forcing_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate predictions autonomously for a given number of steps after
        processing an initial input sequence to set the reservoir state.

        Args:
            initial_input (torch.Tensor): Sequence to process first to set the state
                                          (SeqLen x [BatchSize x] InputDim). The *last*
                                          output prediction or teacher forced value will
                                          be used as the first input for autonomous generation.
            steps (int): Number of future steps to predict autonomously.
            teacher_forcing_targets (Optional[torch.Tensor]): Optional sequence of target values
                                          to use as input for the next step during the prediction
                                          phase, instead of the model's own output. Length should
                                          be at least `steps` if provided fully, or less if only
                                          partially forcing. (Steps x [BatchSize x] OutputDim) -
                                          Note: OutputDim must match InputDim for this simple loop.

        Returns:
            torch.Tensor: Predicted sequence (Steps x [BatchSize x] OutputDim).
        """
        if initial_input.device != self.device:
            initial_input = initial_input.to(self.device)
        if initial_input.dtype != self.dtype:
             print(f"Warning: Initial input dtype ({initial_input.dtype}) differs from model dtype ({self.dtype}). Casting.")
             initial_input = initial_input.to(self.dtype)

        self.eval() # Ensure model is in eval mode

        # Handle optional batch dimension
        batched_input = initial_input.ndim == 3
        if not batched_input:
            initial_input = initial_input.unsqueeze(1) # T x 1 x Dim
        batch_size = initial_input.size(1)

        # --- Warmup Phase: Process the initial_input sequence ---
        with torch.no_grad():
            # Run the initial sequence through the model to set the state
            # The output of this phase is discarded, we only need the final state
            _ = self(initial_input, reset_state=True)

            # Get the *last* actual input used in the warmup phase to start prediction
            # This depends on whether the task is sequence-to-sequence or forecasting
            # Assuming forecasting: use the last element of initial_input if output dim != input dim
            # If output_dim == input_dim, we might want to use the *output* corresponding to the last input.
            # Let's assume output_dim == input_dim for simple prediction loop for now.
            # Get the last predicted output from the warmup phase.
            last_warmup_output = self.readout(self.reservoir_state) # BatchSize x OutputDim

            # Check if output dimension matches input dimension for closed-loop prediction
            if self.output_dim != self.input_dim and teacher_forcing_targets is None:
                 raise ValueError("Output dimension must match input dimension for autonomous prediction "
                                  "without teacher forcing.")

            current_input = last_warmup_output # Start prediction loop with this


        # --- Autonomous Prediction Phase ---
        predictions = []
        with torch.no_grad():
            for step in range(steps):
                # Calculate next state (single step forward)
                # Input: current_input (BatchSize x InputDim)
                # State: self.reservoir_state (BatchSize x ReservoirDim)
                input_term = torch.matmul(current_input, self.W_in.T)
                recurrent_term = torch.matmul(self.reservoir_state, self.W.T)
                # Note: Noise is typically *not* added during prediction
                pre_activation = input_term + recurrent_term
                activated_state = self.activation(pre_activation)
                self.reservoir_state = ((1 - self.leak_rate) * self.reservoir_state +
                                        self.leak_rate * activated_state)

                # Get prediction from the new state
                pred = self.readout(self.reservoir_state) # BatchSize x OutputDim
                predictions.append(pred)

                # Determine input for the *next* step
                if teacher_forcing_targets is not None and step < teacher_forcing_targets.size(0):
                    # Use teacher forcing value if available
                    tf_value = teacher_forcing_targets[step]
                    if not batched_input and tf_value.ndim == 2: # Add batch dim if needed
                         tf_value = tf_value.unsqueeze(0)
                    # Ensure correct shape and device
                    current_input = tf_value.to(device=self.device, dtype=self.dtype)
                    if current_input.shape[-1] != self.input_dim:
                        raise ValueError(f"Teacher forcing target dim ({current_input.shape[-1]}) "
                                         f"doesn't match model input dim ({self.input_dim})")
                else:
                    # Use model's own prediction (requires output_dim == input_dim)
                    if self.output_dim != self.input_dim:
                         # This case was checked earlier, but double-check
                         raise RuntimeError("Cannot use prediction as next input: output_dim != input_dim.")
                    current_input = pred


        # Stack predictions: Steps x BatchSize x OutputDim
        result = torch.stack(predictions, dim=0)

        # Remove batch dimension if input was not batched
        if not batched_input:
            result = result.squeeze(1) # Steps x OutputDim

        return result

    def update_reservoir(self, u: torch.Tensor):
        """
        This method is intended for manual state setting
        If needed.

        """
        print("Warning: `update_reservoir` will update the reservoir states manually. Know what you are doing before using this")
        device = u.device
        self.reservoir_state = u # Directly setting state? Risky.
        self.reservoir_states = torch.cat((self.reservoir_states, self.reservoir_state.unsqueeze(0)), dim=0)
        pass 


    # --- Reservoir Control ---
    def freeze_reservoir(self):
       """Freezes the reservoir weights (W_in, W) so they are not trained."""
       self.W_in.requires_grad = False
       self.W.requires_grad = False
       print("Reservoir weights (W_in, W) frozen.")

    def unfreeze_reservoir(self):
        """Unfreezes the reservoir weights (W_in, W) to allow training (e.g., for finetuning)."""
        self.W_in.requires_grad = True
        self.W.requires_grad = True
        print("Reservoir weights (W_in, W) unfrozen.")

    def reset_state(self, batch_size: int = 1):
        """
        Resets the reservoir's hidden state to zeros.

        Args:
            batch_size (int): The batch size to shape the reset state for.
                              Defaults to 1 if the model hasn't seen batched data yet.
        """
        # Determine batch size - use provided or infer if state exists
        if hasattr(self, 'reservoir_state') and self.reservoir_state.ndim == 2:
            current_batch_size = self.reservoir_state.size(0)
            b_size = batch_size if batch_size > 0 else current_batch_size
        else:
             b_size = batch_size if batch_size > 0 else 1

        self.reservoir_state = torch.zeros(b_size, self.reservoir_dim, device=self.device, dtype=self.dtype)
        self._reservoir_states_list = [] # Clear collected states as well
        print(f"Reservoir state reset for batch size {b_size}.")


    # --- Saving and Loading ---
    def save_model(self, path: str):
        """
        Saves the model's state_dict.

        Args:
            path (str): Path to save the model file.
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str, map_location: Optional[Union[str, torch.device]] = None):
        """
        Loads the model's state_dict.

        Args:
            path (str): Path to the saved model file.
            map_location (Optional[Union[str, torch.device]]): Specifies how to remap storage
                locations (e.g., 'cpu', 'cuda:0'). If None, loads to the locations specified
                in the file. It's often best to load to CPU then move the model.
        """
        if map_location is None:
            map_location = self.device # Load directly to the model's current device by default
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.to(self.device) # Ensure model is on its designated device after loading
        print(f"Model loaded from {path} to device {self.device}")


    # --- Tunnig and Grid Search ---

    def finetune(self,
                 inputs: torch.Tensor,
                 targets: torch.Tensor,
                 epochs: int,
                 lr: float,
                 criterion_class: Type[nn.Module] = nn.MSELoss,
                 optimizer_class: Type[optim.Optimizer] = optim.Adam,
                 print_every: int = 10) -> torch.Tensor:
        """
        Fine-tunes the *entire* model (including reservoir weights W_in, W)
        using backpropagation. Use with caution, as this deviates from the, device=Model.device())
        standard ESN fixed-reservoir principle. Ensure reservoir is unfrozen.

        Args:
            inputs (torch.Tensor): Input sequence (SequenceLength x [BatchSize x] InputDim).
            targets (torch.Tensor): Target sequence (SequenceLength x [BatchSize x] OutputDim).
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
            criterion_class (Type[nn.Module]): Loss function class (default: MSELoss).
            optimizer_class (Type[optim.Optimizer]): Optimizer class (default: Adam).
            print_every (int): Frequency of printing loss updates.

        Returns:
            torch.Tensor: Tensor containing the loss value for each epoch.
        """
        if not self.W_in.requires_grad or not self.W.requires_grad:
             print("Warning: Finetuning called, but reservoir weights (W_in, W) are frozen. "
                   "Call Unfreeze_reservoir() first if you intend to train them.")

        if inputs.device != self.device or targets.device != self.device:
             raise ValueError(f"Input/Target tensor devices ({inputs.device}/{targets.device}) must match model device ({self.device}).")
        if inputs.dtype != self.dtype or targets.dtype != self.dtype:
             print(f"Warning: Input/Target dtypes ({inputs.dtype}/{targets.dtype}) differ from model dtype ({self.dtype}). Casting.")
             inputs = inputs.to(self.dtype)
             targets = targets.to(self.dtype)

        # Define loss function and optimizer
        criterion = criterion_class()
        optimizer = optimizer_class(self.parameters(), lr=lr) # self.parameters() includes W_in, W if requires_grad=True

        losses = torch.zeros(epochs, device=self.device) # Pre-allocate losses tensor

        self.train() # Set model to training mode

        for epoch in range(epochs):
            optimizer.zero_grad()
            # Reset state for each epoch pass? Depends on task. Assume yes for typical sequence tasks.
            output = self(inputs, reset_state=True)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            losses[epoch] = loss.item() # Store loss

            if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
                print(f'Finetune Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}')

        self.eval() # Set model back to evaluation mode
        return losses


    def best_hyperparameters():
        pass