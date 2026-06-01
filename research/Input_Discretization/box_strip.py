import os
import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted

# Add your project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from reservoirgrid.helpers import chaos_utils, utils
from joblib import Parallel, delayed

def process_file(file, system_path):
    file_path = os.path.join(system_path, file + ".pkl")
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        results = []
        for entry in data:
            try:
                lyap = chaos_utils.lyapunov_time(
                    truth=entry['true_value'],
                    predictions=entry["predictions"]
                )
                kldiv = chaos_utils.kl_divergence(
                    truth=entry['true_value'],
                    predictions=entry["predictions"]
                )
                results.append((
                    entry['parameters']['LeakyRate'],
                    entry['parameters']['SpectralRadius'],
                    entry['parameters']['InputScaling'],
                    lyap,
                    kldiv,
                    file
                ))
            except KeyError as e:
                print(f"KeyError in {file}: {e}")
                continue
        return results
    except Exception as e:
        print(f"Error in {file}: {e}")
        return []

def plot_box_strip(df, system_name, save_path, x_var='LeakyRate', y_var='KLDivergence', log_scale = True):
    """
    Create box and strip plots for the given DataFrame
    
    Parameters:
    df (pd.DataFrame): Data to plot
    system_name (str): Name of the system for title
    save_path (str): Path to save the figure
    x_var (str): Variable for x-axis (default: 'LeakyRate')
    y_var (str): Variable for y-axis (default: 'KLDivergence')
    """
    plt.figure(figsize=(12, 8))
    
    # Create FacetGrid
    g = sns.FacetGrid(df, col='File', col_wrap=5, height=3, sharey=False)
    
    # Add boxplots
    g.map(sns.boxplot,
          x_var,
          y_var,
          color='lightgray',
          width=0.6,
          order=df[x_var].unique(),
          showfliers=False,
          whiskerprops={'linewidth': 2},
          boxprops={'facecolor': 'lightgray', 'edgecolor': 'black'},
          medianprops={'color': 'green', 'linewidth': 2})
    
    # Add stripplots
    g.map(sns.stripplot, 
          x_var, 
          y_var,    
          hue=df[x_var],
          palette=sns.color_palette("husl", n_colors=len(df[x_var].unique())),
          jitter=True,
          order=df[x_var].unique(),
          size=6,
          alpha=0.5,
          linewidth=0.5,
          edgecolor='white',
          dodge=False)
    
    # Set logarithmic scale for y-axis if enabled
    if log_scale:
        for ax in g.axes.flat:
            ax.set_yscale('log')
    
    # Set titles and save
    g.set_titles("Point Per Period - {col_name}")
    g.fig.suptitle(f"{system_name} - {y_var} vs {x_var}")
    plt.tight_layout()
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{system_name}_{y_var}_{x_var}.png", dpi=600)
    plt.close()

# Main processing
path = "Examples/Input_Discretization/results/"  # path to directory of system class
types = os.listdir(path)  # Stores types of system

for system_type in types:
    if system_type == "test":
        continue  # Skip test directory

    system_list = os.listdir(path + system_type)  # list of systems, chen, chua, rossler etc
    for system in system_list:
        system_path = os.path.join(path, system_type, system)
        print(f"selected system: {system_path}")
        all_files = os.listdir(system_path)
        
        # Get sorted list of files (without .pkl extension)
        sorted_files = natsorted([name[:-4] for name in all_files if name.endswith('.pkl')])
        print(f"Calculating Results...")
        # Process all files in parallel
        results = Parallel(n_jobs=-1)(
            delayed(process_file)(file, system_path) for file in sorted_files
        )
        print(f"Calculated Results")
        # Flatten results
        flat_results = [item for sublist in results for item in sublist] # type: ignore
        
        # Create DataFrame
        df = pd.DataFrame(flat_results, columns=[
            'LeakyRate', 'SpectralRadius', 'InputScaling', 
            'LyapunovTime', 'KLDivergence', 'File'
        ])
        
        # Create plots for different parameter combinations
        save_path = os.path.join("Examples/Input_Discretization/Plots/SingleMetric", system_type, system)
        
        print(f"Saving Plots...")
        plot_box_strip(df, system, save_path, x_var='SpectralRadius', y_var='LyapunovTime', log_scale=True)
        plot_box_strip(df, system, save_path, x_var='LeakyRate', y_var='LyapunovTime', log_scale = True)
        plot_box_strip(df, system, save_path, x_var='InputScaling', y_var='LyapunovTime', log_scale = True)
        print(f"Plots Saved! for system {system}")
        