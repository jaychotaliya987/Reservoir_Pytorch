from timeit import default_timer as timer

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import numpy as np
from dysts.flows import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

from reservoirgrid.models import Reservoir
from reservoirgrid.datasets import LorenzAttractor
from reservoirgrid.helpers import utils, viz, reservoir_tests


system_name = "Lorenz"
path = "reservoirgrid/datasets/Chaotic/" + system_name + ".npy"


if not os.path.exists(path):
    print("System does not exist, Generate First")
else:
    print("System exist, loading from datasets")
    system = np.load(path, allow_pickle=True)
    print("System loaded")

system = utils.truncate(system)
input = system[2][1]
input = utils.normalize_data(input)

parameter_dict = {
    "SpectralRadius": [0.7, 0.9, 1.1],
    "LeakyRate": [0.3, 0.5, 0.7],
    "InputScaling": [0.3, 0.5, 1.0]
}

results = utils.parameter_sweep(inputs=input, parameter_dict=parameter_dict, 
                        reservoir_dim=1300, input_dim= 3, 
                        output_dim=3, sparsity=0.9)

print(results)
