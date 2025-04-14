import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

matplotlib.use('qt5Agg')  # Use a non-interactive backend for saving plots

# Ensure the correct path to MackeyGlass module
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from Models.Reservoir import Reservoir
from Datasets.LorenzAttractor import LorenzAttractor as LAttractor

print ("Imports done! \n")

attractor = LAttractor(
    sample_len=1000,
    n_samples=1,
    xyz=(0.0, 1.0, 1.05),   # initial condition
    sigma=10.0,             # default Lorenz sigma
    b=8/3,                  # default Lorenz b (beta)
    r=28.0                  # default Lorenz r (rho)
)

print(attractor)
