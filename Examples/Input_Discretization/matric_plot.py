import os
import sys

import gc
import tracemalloc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from reservoirgrid.models import Reservoir
from reservoirgrid.helpers import chaos_utils, utils, viz, reservoir_tests
from matplotlib.colors import LogNorm, Normalize

import matplotlib.pyplot as plt
import numpy as np
import pickle

path = "Examples/Input_Discretization/results/Chaotic/Lorenz/"

file_path = path + "10.0.pkl"
with open(file_path, 'rb') as f:
    data = pickle.load(f)

for entry in data:

    leaky = entry['parameters']['LeakyRate']
    lyap_time = chaos_utils.comparative_lyapunov_time(
                test_targets = entry['true_value'],
                predictions = entry["predictions"]
    )
    
