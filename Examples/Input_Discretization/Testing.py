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
file = "Lorenz/" + "100.0.pkl"
file_path = path + file

if not os.path.isfile(file_path):
    print("file does not exist")
    

with open(file_path, "rb") as f:
    results = pickle.load(f)

viz.plot_multidimensional_3d(results, "Lorenz", save_html = True, pp = 100)
