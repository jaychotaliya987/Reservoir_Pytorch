from timeit import default_timer as timer
import gc
import tracemalloc

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


#Parameter Dictionaries
parameter_dict = {
    "SpectralRadius": [0.7, 0.8, 0.9, 1.0, 1.1, 1.],
    "LeakyRate": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
    "InputScaling": [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
}

parameter_dict_single = {
    "SpectralRadius": [1],
    "LeakyRate": [0.5],
    "InputScaling": [0.5]
}

# System List
system_list = ["MultiChua", "Rossler"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loop Through System_list
for system in system_list:
    system_name = "/Chaotic/" + system
    print(f"selected {system_name}")
    path = "../../reservoirgrid/datasets" + system_name + ".npy"

    if not os.path.exists(path):
        print("System does not exist, Generate First")
        exit()
    else:
        print("System exist, loading from datasets")
        system = np.load(path, allow_pickle=True)
        print("System loaded")

    T_system = utils.truncate(system) #truncated system to have same periods

    #loop to calculate whole system with the parameter dict. Calculates 20 system with  
    for pp_select in range(len(T_system['pp'])):
        print(f"selected point per periods: {T_system[pp_select][0]}")
        start = timer()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        input = T_system[pp_select][1] # select inputs from truncated system
        input = utils.normalize_data(input) # Normalizes Inputs
        
        r_dim = input.shape[1] # set the input output dims of the reservoir from the input's dimension
        
        results = utils.parameter_sweep(inputs=input, parameter_dict=parameter_dict, 
                            reservoir_dim=1300, input_dim= r_dim, 
                           output_dim=r_dim, sparsity=0.9, return_targets=True)


        pp_num = str(T_system[pp_select][0]) #extract pp_num for name saving
        result_folder =  "results" + system_name
        result_path = "results" + system_name + "/" + pp_num + ".pkl" # path of results
        
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
            
        with open(result_path , 'wb') as f:
            pickle.dump(results, f)

        #releases memory of results, already saved to hard-drive
        del results
        gc.collect()

        end = timer()
        snapshot2 = tracemalloc.take_snapshot()
        stats = snapshot2.compare_to(snapshot1, 'lineno') # calculates the freed memory

        print(f"Memory released by deleting results: {stats[0].size_diff / 10**6:.2f} MB")
        print(f"loop time {end - start:.4f} seconds\n")
