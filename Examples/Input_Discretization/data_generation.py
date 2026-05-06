#------------------ General Purpose Imports ---------------------#
import pandas as pd
from timeit import default_timer as timer

#------------------ Dataset imports ---------------------#
from dysts.flows import *
from dysts import flows

#------------------ system imports ---------------------#
import os
import sys
sys.path.append(os.path.abspath(os.path.join('../..')))

#------------------ reservoirgrid imports ---------------------#
from reservoirgrid.helpers import utils
#--------------------------------------------------------------#


# Selecting the system/type of system
systems = pd.read_csv("../../reservoirgrid/datasets/systems.csv")
RegimeSwitching = systems[systems['Type']=='Chaotic']


point_per_period = np.linspace(start=5, stop=100, num= 20) # discretizing 20 times from 0 PPP to 100 PPP

for index, row in RegimeSwitching.iterrows():
    system_name = row['System']
    system_type = row['Type']
    path = f"../../reservoirgrid/datasets/{system_type}/{system_name}.npy"

    if not os.path.exists(path):
        print(f"{system_name} does not exist. Generating...")

        try:
            # Dynamically get system class from dysts.flows
            system = getattr(flows, system_name)

            start = timer()
            data = utils.discretization(system, point_per_period, trajectory_length=10000) ## Remember to change the discretization to discretization_with_dt if you want high accuracy in the base system.
            end = timer()

            print(f"Generation Time: {end - start:.4f} seconds")

            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, data)
        except AttributeError:
            print(f" {system_name} not found in dysts.flows.")
    else:
        print(f"{system_name} exists.")
