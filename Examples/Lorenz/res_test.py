#------------------ General Purpose Imports ---------------------#
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import plotly.graph_objects as go

#------------------ Machine Learning Imports ---------------------#
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split

#------------------ system imports ---------------------#
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

#------------------ reservoirgrid imports ---------------------#
from reservoirgrid.models import Reservoir
from reservoirgrid.helpers import utils
from reservoirgrid.helpers import viz
from reservoirgrid.helpers import chaos_utils
from reservoirgrid.helpers import reservoir_tests
#--------------------------------------------------------------#

system_name = "/Chaotic/" + "Rossler"

print(f"selected {system_name}")
path = "reservoirgrid/datasets" + system_name + ".npy"
if not os.path.exists(path):
    print("System does not exist, Generate First")
    exit()
else:
    print("System exist, loading from datasets")
    system = np.load(path, allow_pickle=True)
    print("System loaded")

T_system = utils.truncate(system) #truncated system to have same periods
input = T_system[10][1] # select inputs from truncated system
input = utils.normalize_data(input) # Normalizes Inputs

train_inputs, test_inputs, train_targets, test_targets = utils.split(input)

ResRose = Reservoir(
    input_dim=3,
    reservoir_dim=1300,
    output_dim=3,
    spectral_radius=1,
    leak_rate=0.5,
    sparsity=0.9,
    input_scaling=0.5,
    noise_level = 0.01
)

ResRose.train_readout(train_inputs, train_targets, warmup=10)
states = ResRose.reservoir_states.cpu()
viz.visualize_reservoir_states(model = ResRose, show_distribution=False)

prediction = ResRose.predict(train_inputs, steps=len(test_targets)).cpu()
rmse = utils.RMSE(test_targets, prediction)
print(f"RMSE: {rmse}")
