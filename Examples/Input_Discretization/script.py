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

system_name = "name"
path = "reservoirgrid/datasets/" + system_name + ".npy"
save_path = "reservoirgrid/datasets/Chaotic/HyperLorenz.npy"

if not os.path.exists(save_path):
    print("System does not exist, Generating...")
    times = np.linspace(start=5, stop=100, num= 20)
    start = timer()
    system = utils.discretization(Lorenz, times, trajectory_length= 10000)
    end = timer()
    print(f"Generation Time: {end - start:.4f} seconds")
    np.save(save_path, system)
else:
    print("System exist, loading from dataset")
    system = np.load(save_path, allow_pickle=True)
    print("System loaded")

attractor1 = system[0]
attractor2 = system[1]

print(system[1][1])


viz.compare_plot([system[1][1][:500],system[2][1]][:500])
viz.plot_components(system[1][1][:500])
viz.plot_components(system[2][1][:500])
plt.show()

