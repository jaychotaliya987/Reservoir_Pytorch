import plotly.graph_objects as go
import numpy as np
import torch
from Datasets.LorenzAttractor import LorenzAttractor

# Generate the Lorenz Attractor data
attractor = LorenzAttractor(sample_len=10000, n_samples=10, xyz=[1.0, 1.0, 1.0], sigma=10.0, b=8/3, r=28.0)
attractor_samp = attractor[1].numpy()  # Convert to numpy array


