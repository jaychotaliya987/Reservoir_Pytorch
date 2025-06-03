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


save_path = "reservoirgrid/datasets/Lorenz.npy"

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

#viz.compare_plot(datasets=system[0][1], titles=system[0][0])
viz.plot_components(attractor1[1], labels=["X","Y","Z"], title=attractor1[0],linewidth=0.1)
#viz.plot_components(attractor2[1], labels=["X","Y","Z"])

plt.show()
