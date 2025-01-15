import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure the correct path to MackeyGlass module
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from Datasets.MackeyGlass2DDataset import MackeyGlass2DDataset
from Datasets.MackeyGlassDataset import MackeyGlassDataset
from Datasets.MackeyGlass import MackeyGlass
from Reservoir import Echostate

from torch.utils.data import DataLoader

print("Imports Done!")


Mglass1 = MackeyGlass2DDataset(20, 100, 17, 1)

Echostate1 = Echostate.EchoStates(1, 100, num_layers=1, batch_first=True)
