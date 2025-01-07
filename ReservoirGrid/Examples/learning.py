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



print("Imports Done!")

Mglass1 = MackeyGlass2DDataset(20, 100, 17, 1)
Mglass2 = MackeyGlassDataset(20, 100)



print(Mglass1.__len__())
print(Mglass1.__getitem__(1))



