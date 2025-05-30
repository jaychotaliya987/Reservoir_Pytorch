from timeit import default_timer as timer

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import numpy as np
import dysts
from dysts.systems import make_trajectory_ensemble

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

from reservoirgrid.models import Reservoir
from reservoirgrid.datasets import LorenzAttractor
from reservoirgrid.helpers import utils
from reservoirgrid.helpers import reservoir_tests


all_out = make_trajectory_ensemble(100, resample=True, pts_per_period=75)




