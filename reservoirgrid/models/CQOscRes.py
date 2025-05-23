import torch 
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Optional, Callable, Type, Union
import qutip as qt

# Default device (can be overridden)
_DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DEFAULT_DTYPE = torch.float32

class CQOscRes(nn.Module):
    def __init__(self,
            eps_0: float,
            input_dim: int,
            omega: tuple,
            kappa: tuple,
             coupling: float,
            output_dim: int,
            time: float,
            inference: int,
            h_truncate: int = 8, 
            device: Optional[Union[str, torch.device]] = None,
            dtype: torch.dtype = _DEFAULT_DTYPE):
        """
        Initializes the coupled harmonic oscillator reservoir.
        args:
            :param eps_0 :  
        """
        super().__init__()

        self.eps_0 = eps_0
        self.input_dim = input_dim
        self.h_truncate = h_truncate
        self.omega = omega
        self.kappa = kappa
        self.coupling = coupling
        self.time = np.linspace(0, time, inference)
        self.dtype = dtype
        self.device = torch.device(device) if device else _DEFAULT_DEVICE

        # Quantum operators
        self.a = qt.tensor(qt.destroy(h_truncate), qt.qeye(h_truncate))
        self.b = qt.tensor(qt.qeye(h_truncate), qt.destroy(h_truncate))
        
        # Hamiltonian components
        self.H_static = (self.omega[0] * self.a.dag() * self.a +
                        self.omega[1] * self.b.dag() * self.b +
                        self.coupling * (self.a * self.b.dag() + self.a.dag() * self.b))

        # Quantum state storage (density matrices)
        self.state_dim = h_truncate**2  # For 2 oscillators
        self.state_list = torch.zeros((self.state_dim, self.state_dim, len(self.time)),
                                    dtype=dtype, device=device)

        # Readout layer
        self.readout = nn.Linear(self.state_dim * 2, output_dim,  # Real + Imag parts
                               device=self.device, dtype=self.dtype)

    def forward(self, u: torch.Tensor, p_shots: Optional[int] = None) -> torch.Tensor:
        # Convert input to numpy array and interpolate
        u_np = u.detach().cpu().numpy()
        time_points = self.time

        # Time-dependent Hamiltonian coefficient function
        def H_drive_coeff(t, args):
            index = np.clip(np.searchsorted(time_points, t), 0, len(u_np)-1)
            return u_np[index]

        H_drive = [self.eps_0 * np.sqrt(2 * self.kappa[0]) * (self.a + self.a.dag()), 
                 H_drive_coeff]

        # Collapse operators
        c_ops = [np.sqrt(self.kappa[0]) * self.a, 
                np.sqrt(self.kappa[1]) * self.b]

        # Initial state
        psi0 = qt.tensor(qt.basis(self.h_truncate, 0), 
                       qt.basis(self.h_truncate, 0))

        # Solve master equation
        result = qt.mesolve([self.H_static, H_drive], psi0, self.time, c_ops, [])

        # Process density matrices
        states = []
        for i, rho in enumerate(result.states):
            # Convert to complex PyTorch tensor
            rho_np = rho.full()
            rho_tensor = torch.tensor(rho_np, dtype=torch.complex64, device=self.device)
            self.state_list[:, :, i] = rho_tensor

            # Separate real and imaginary parts
            state_real = torch.view_as_real(rho_tensor).flatten()  # [N, N, 2]
            states.append(state_real.float())  # Convert complex to real/imag pairs

        # Apply readout layer
        self.states = torch.stack(states)
        #self.states = self.readout(self.states)
        return self.states

    def measure_p_shots(self, rho: qt.Qobj, observable: qt.Qobj, p_shots: int = 100) -> float:
        """Simulate P-shot measurements of an observable"""
        # Get eigenvalues and projectors
        eigvals, eigstates = observable.eigenstates()

        # Calculate probabilities
        probs = np.array([(e.dag() * rho * e).tr().real for e in eigstates])
        probs = np.maximum(probs, 0)  # Ensure non-negative
        probs /= probs.sum()  # Normalize

        # Simulate measurements
        results = np.random.choice(eigvals, p_shots, p=probs)
        return results.mean()

    def train_readout(self, X: torch.Tensor, y: torch.Tensor, alpha: float = 1e-4):
        """Ridge regression for readout training"""
        # Convert to numpy for closed-form solution
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
        
        print(X_np.shape)
        # Ridge regression solution
        I = np.eye(len(X_np))
        theta = np.linalg.inv(X_np.T @ X_np + alpha * I) @ X_np.T @ y_np
        
        # Update readout weights
        with torch.no_grad():
            self.readout.weight.data = torch.tensor(theta.T, dtype=self.dtype, device=self.device)
            self.readout.bias.data.zero_()

    def verify_density_matrix(self):
        """Check density matrix properties"""
        for i in range(len(self.time)):
            rho = self.state_list[:, :, i].cpu().numpy()
            trace = np.trace(rho)
            eigvals = np.linalg.eigvalsh(rho)
            print(f"Time {self.time[i]:.2f}: Trace={trace:.4f}, Min Eigenvalue={eigvals[0]:.2e}")

    def classify(self, tesesequance):
        """Make predictions using the trained readout layer"""
        classification_list = []
        testseqance = 123 
        return classification_list
    
    def RMSE(self, y_true, y_pred):
        """Calculate Root Mean Square Error"""
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
