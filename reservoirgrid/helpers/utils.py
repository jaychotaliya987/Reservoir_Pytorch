from timeit import default_timer as timer

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from reservoirgrid.models import Reservoir
from reservoirgrid.datasets import LorenzAttractor
#from dysts.datasets import load_dataset

def normalize_data(data):
    """
    Normalize data while preserving system dynamics, 
    and centers the data in the range [-1,1]
    """
    return (data - data.min()) / (data.max() - data.min()) *2 -1

def discretization(data, points):
    pass