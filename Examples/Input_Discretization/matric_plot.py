import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from reservoirgrid.models import Reservoir
from reservoirgrid.helpers import chaos_utils, utils, viz, reservoir_tests
from matplotlib.colors import LogNorm, Normalize

import gc
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
import pickle


path = "Examples/Input_Discretization/results/Quasiperiodic/"
save_path = "Examples/Input_Discretization/Plots/SingleMetric/"
system_name = "Torus"
system_path = os.path.join(path, system_name)

# Create a color map for different files
cmap = plt.get_cmap('tab10') 
all_leaky = []
all_lyap = []
file_labels = []

plt.figure(figsize=(12, 7))

for i, file in enumerate(os.listdir(system_path)):
    file_path = os.path.join(system_path, file)
    
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    # Initialize arrays for this file
    leaky_rates = []
    lyap_times = []
    
    for entry in data:
        try:
            leaky = entry['parameters']['LeakyRate']
            lyap_time = chaos_utils.comparative_lyapunov_time(
                        test_targets=entry['true_value'],
                        predictions=entry["predictions"]
            )
            leaky_rates.append(leaky)
            lyap_times.append(lyap_time)
        except Exception as e:
            print(f"Error processing entry in {file}: {e}")
            continue

    all_leaky.extend(leaky_rates)
    all_lyap.extend(lyap_times)
    file_labels.extend([file]*len(leaky_rates))
    

    plt.scatter(lyap_times, leaky_rates, 
               color=cmap(i % 10),  # Cycle through 10 colors
               alpha=0.7,
               label=f'{file[:-6]}')

plt.ylabel('Leaky Rate', fontsize=12)
plt.xlabel('Comparative Lyapunov Time', fontsize=12)
plt.title(f'{system_name} System: Leaky Rate vs. Lyapunov Time', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig(f"{save_path}{system_name}_Leaky_Lyp", dpi = 600)
plt.show()