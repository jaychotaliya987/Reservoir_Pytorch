import numpy as np
import scipy
import torch
import pickle

import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from reservoirgrid.models import Reservoir
from reservoirgrid.helpers import utils, viz, reservoir_tests, chaos_utils

path = "Examples/Input_Discretization/results/Chaotic/"
system_name = "Chua"
system_path = path + system_name
   
for file in os.listdir(system_path):
    file_path = system_path + "/" + file
     
    with open(file_path, "rb") as f:
        results = pickle.load(f)
    
    viz.plot_multidimensional_3d(results, system_name, save_html = True, pp = int(file[:-6]))

    
