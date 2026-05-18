import numpy as np
import scipy
import torch
import pickle

import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from reservoirgrid.models import Reservoir
from reservoirgrid.helpers import utils, viz, reservoir_tests, chaos_utils

path = "Examples\\Input_Discretization\\results\\Chaotic\\"
system_name = "LorenzLHS"
system_path = path + system_name

if not os.path.exists("Examples/Input_Discretization/Plots/3DPlots/" + system_name):
    os.makedirs("Examples/Input_Discretization/Plots/3DPlots/"+ system_name)

file_path = system_path + "/" + "75.0.pkl"

with open(file_path, "rb") as f:
    results = pickle.load(f)

metrics_dict = {
    "KL Divergence": [chaos_utils.kl_divergence(r["true_value"], r["predictions"]) for r in results],
    "JS Divergence": [chaos_utils.js_divergence(r["true_value"], r["predictions"]) for r in results],
    "RMSE": [np.sqrt(np.mean((r["true_value"] - r["predictions"]) ** 2)) for r in results]
}

viz.plot_multidimensional_3d(results, system_name, pp = 75, metrics_dict = metrics_dict).show()


    
# for file in os.listdir(system_path):
#     file_path = system_path + "/" + file
     
#     with open(file_path, "rb") as f:
#         results = pickle.load(f)
    
#     viz.plot_multidimensional_3d(results, system_name, save_html = True)
#     print(file)
#     break
