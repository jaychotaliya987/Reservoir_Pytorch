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

path = "Examples/Input_Discretization/results/Chaotic/"

for folder_name in os.listdir(path):
    result_path = "Examples/Input_Discretization/results/Quasiperiodic/" + "CaTwoPlus"
    print(result_path)

# Initialize storage
    all_data = []

# Load all data
    for file_name in os.listdir(result_path):
        if not file_name.endswith('.pkl'):
            continue
        
        ppp = float(file_name[:-4])  # Extract points-per-period
        file_path = os.path.join(result_path, file_name)

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        for entry in data:
            params = entry['parameters']
            lyap_time = chaos_utils.comparative_lyapunov_time(
                test_targets=entry['true_value'],
                predictions=entry['predictions']
            )
            kldiv = chaos_utils.KLdivergence(entry['true_value'], entry['predictions'], bins = 100)
            psd, cos_sim = chaos_utils.psd_errors(entry['true_value'], entry['predictions'], return_cos_sim=True)

            all_data.append({
                'ppp': ppp,
                'params': params,  # Store the full parameter dictionary
                'LyapunovTime': lyap_time,
                'KLDivergence': kldiv,
                'PSD Errors' : psd,
                'Cos_Sim': cos_sim,
                'RMSE': entry['metrics']['RMSE']
            })
        del data
        gc.collect()


    # DataFrame for easier manipulation
    import pandas as pd
    df = pd.DataFrame(all_data)
    del all_data
    gc.collect()
    
    # identifiers for each parameter combination
    df['param_combo'] = df['params'].apply(lambda x: tuple(sorted(x.items())))

    # Get sorted unique values
    unique_ppp = np.sort(df['ppp'].unique())
    unique_param_combos = sorted(df['param_combo'].unique())

    #Metrics to plot
    metrics = ['LyapunovTime', 'KLDivergence', 'PSD Errors', "RMSE", 'Cos_Sim']  

    for metric in metrics:
        # heatmap matrix for current metric:
        heatmap = np.full((len(unique_ppp), len(unique_param_combos)), np.nan)

        for i, ppp in enumerate(unique_ppp):
            for j, combo in enumerate(unique_param_combos):
                match = df[(df['ppp'] == ppp) & (df['param_combo'] == combo)]
                if not match.empty:
                    heatmap[i, j] = match[metric].values[0]

        # Plotting
        plt.figure(figsize=(20, 8))
        plt.imshow(heatmap, 
                   cmap='viridis',
                   aspect='auto',
                   origin='lower',
                   interpolation='nearest',
                   norm =(LogNorm() if metric in ["PSD Errors", "RMSE"] 
                            else Normalize() if metric == "Cos_Sim" 
                            else None))

        # Customize plot for metric
        xticks = range(0, len(unique_param_combos), 5)
        xlabels=[f"{i}" for i in xticks]
        plt.xticks(ticks=xticks, labels=xlabels, rotation=90)
        plt.ylabel('Points per Period')
        plt.colorbar(label=metric)
        plt.title(f'{metric} vs Parameter Combinations and Points-per-Period')
        plt.tight_layout()

        # Save with metric-specific filename
        plt.savefig(f"{result_path}{metric}.png", dpi=600)
        plt.close() 
