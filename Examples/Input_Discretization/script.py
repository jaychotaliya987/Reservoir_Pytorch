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
from reservoirgrid.helpers import utils
from reservoirgrid.helpers import viz
from reservoirgrid.helpers import reservoir_tests


attractor1 = Lorenz().make_trajectory(n=100, pts_per_period=50, return_times= True)
attractor2 = Lorenz().make_trajectory(n=100, pts_per_period=10, return_times= True)

attractors = [attractor1[1], attractor2[1]]

viz.compare_plot(datasets=attractors, titles=["0.01", "0.02"], figsize=(1920,1080))
viz.plot_components(attractor1[1], labels=["X","Y","Z"])
viz.plot_components(attractor2[1], labels=["X","Y","Z"])

