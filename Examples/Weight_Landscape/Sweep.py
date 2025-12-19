from timeit import default_timer as timer
import gc
import tracemalloc

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
import pickle

from reservoirgrid.helpers import utils
from scipy.stats import qmc

# --- CONFIGURATION ---
# Set this to the specific PP value you want to run (e.g., 75, 100, etc.)
TARGET_PP = 75 
# ---------------------

# System List
system_list = ["Lorenz"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Setup
d = 3
n = 500

# 2. Generate raw samples (0.0 to 1.0)
sampler = qmc.LatinHypercube(d=d)
sample_01 = sampler.random(n=n)

# 3. Define Manual Bounds [SpectralRadius, LeakyRate, InputScaling]
# Lower limits for your 3 variables
l_bounds = [0.7, 0.1, 0.2]  
# Upper limits for your 3 variables
u_bounds = [1.1, 0.95, 1.0] 

# 4. Scale the samples to these bounds
scaled_sample = np.round(qmc.scale(sample_01, l_bounds, u_bounds), 2)

# 5. Create your dictionary manually
parameter_dict = {
    "SpectralRadius": scaled_sample[:, 0],
    "LeakyRate":      scaled_sample[:, 1],
    "InputScaling":   scaled_sample[:, 2]
}

# Loop Through System_list
for system in system_list:
    system_name = "/Chaotic/" + system
    print(f"selected {system_name}")
    path = "reservoirgrid/datasets" + system_name + ".npy"

    if not os.path.exists(path):
        print("System does not exist, Generate First")
        exit()
    else:
        print("System exist, loading from datasets")
        system_data = np.load(path, allow_pickle=True) # Renamed to system_data to avoid masking 'system' var
        print("System loaded")

    T_system = utils.truncate(system_data) #truncated system to have same periods

    # --- MODIFICATION: Find the specific index for TARGET_PP ---
    selected_indices = []
    
    # Search T_system to find the index where the PP matches TARGET_PP
    # We assume T_system['pp'] defines the number of entries, and T_system[i][0] is the PP value
    for i in range(len(T_system['pp'])):
        if T_system[i][0] == TARGET_PP:
            selected_indices.append(i)
            break # Stop after finding the first match
            
    if not selected_indices:
        print(f"Warning: TARGET_PP {TARGET_PP} not found in system {system_name}. Skipping.")
        continue
    
    print(f"Found TARGET_PP {TARGET_PP} at index {selected_indices[0]}. Processing...")

    # Run loop ONLY for the selected index
    for pp_select in selected_indices:
        print(f"selected point per periods: {T_system[pp_select][0]}")
        start = timer()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        input = T_system[pp_select][1] # select inputs from truncated system
        input = utils.normalize_data(input) # Normalizes Inputs
        
        r_dim = input.shape[1] # set the input output dims of the reservoir from the input's dimension
        
        results = utils.parameter_sweep_comp(inputs=input, parameter_dict=parameter_dict, 
                                    reservoir_dim=1300, input_dim= r_dim, 
                                   output_dim=r_dim, sparsity=0.9, return_targets=True, sampled_dict=True)


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