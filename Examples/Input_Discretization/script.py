from timeit import default_timer as timer

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import numpy as np
from dysts.flows import *
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

from reservoirgrid.models import Reservoir
from reservoirgrid.datasets import LorenzAttractor
from reservoirgrid.helpers import utils, viz, reservoir_tests


system_name = "Lorenz"
path = "../../reservoirgrid/datasets/Chaotic/" + system_name + ".npy"

if not os.path.exists(path):
    print("System does not exist, Generate First")
else:
    print("System exist, loading from datasets")
    system = np.load(path, allow_pickle=True)
    print("System loaded")

T_system = utils.truncate(system) #truncated system to have same periods
pp_select = 5

input = T_system[pp_select][1] # selecting the sample 
input = utils.normalize_data(input)

parameter_dict = {
    "SpectralRadius": [0.7, 0.8, 0.9, 1.0, 1.1, 1.],
    "LeakyRate": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
    "InputScaling": [0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
}

parameter_dict1 = {
    "SpectralRadius": [0.7, 0.8, 0.9],
    "LeakyRate": [0.1, 0.3, 0.5],
    "InputScaling": [0.05, 0.2, 0.4]
}

parameter_dict2 = {
    "SpectralRadius": [1.0, 1.1, 1.0],
    "LeakyRate": [0.7, 0.9, 0.95],
    "InputScaling": [0.5, 0.6, 0.8, 1.0]
}

parameter_dict_single = {
    "SpectralRadius": [1],
    "LeakyRate": [0.5],
    "InputScaling": [0.5]
}

results = utils.parameter_sweep(inputs=input, parameter_dict=parameter_dict1, 
                        reservoir_dim=1300, input_dim= 3, 
                       output_dim=3, sparsity=0.9, return_targets=True)

pp_num = str(T_system[pp_select][0])
result_path = "results/" + system_name + "/" + pp_num + "_dict1" + ".pkl"

with open(result_path , 'wb') as f:
    pickle.dump(results, f)
