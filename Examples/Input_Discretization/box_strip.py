import os, sys, gc, tracemalloc
import pickle

import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from reservoirgrid.models import Reservoir
from reservoirgrid.helpers import chaos_utils, utils, viz, reservoir_tests


path = "Examples/Input_Discretization/results/" # path to directory of system class
types = os.listdir(path) # Stores types of system

for system_type in types:
    if system_type == "test": break ## Break Statement for not to pull data from test directory

    system_list = os.listdir(path + system_type) # list of systems, chen, chua, rossler etc
    for system in system_list:

        all_files = os.listdir(path + system_type + "/" + system)
        files_to_plot = all_files[:]
        sort = natsorted([name[:-4] for name in files_to_plot])
        
        pass
    

