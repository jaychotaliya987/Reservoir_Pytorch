from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import torch
import random

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reservoirgrid.models import Reservoir
from reservoirgrid.helpers import chaos_utils
from reservoirgrid.helpers import utils
from reservoirgrid.helpers import viz


path = "Examples/Weight_Landscape/results/Chaotic/"
save_path = "Examples/Weight_Landscape/Plots/SingleMetric/"
system_name = "Lorenz"
system_path = os.path.join(path, system_name)

file = os.path.join(system_path, "75.0.pkl")
with open(file, "rb") as f:
    data_10 = pickle.load(f)


all_lyapunov = []
all_kldiv = []
all_params = []
all_jsdiv = []
all_skl = []

for data in data_10:
    lyap1 = chaos_utils.lyapunov_time(data["true_value"], data["predictions"])
    Kldiv = chaos_utils.kl_divergence(data["true_value"], data["predictions"])
    jsdiv = chaos_utils.js_divergence(data["true_value"], data["predictions"])
    skl = chaos_utils.symmetric_kl(data["true_value"], data["predictions"])

    all_skl.append(skl)
    all_jsdiv.append(jsdiv)
    all_lyapunov.append(lyap1)
    all_kldiv.append(Kldiv)
    
    params = data["parameters"]
    all_params.append(params)


jsidx = np.argpartition(all_jsdiv, 10)[:10]
klidx = np.argpartition(all_kldiv, 10)[:10]
sklidx = np.argpartition(all_skl, 10)[:10]


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# define grid shape
n = len(jsidx)
cols = 5   # needs atleast 10 values to se how the kl divergence favours the coverage of base attractor over the perfect coverage
rows = int(np.ceil(n / cols))

clean_scene = dict(
    camera=dict(
        eye=dict(x=-1.5, y=1.5, z=0.5),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
    ),
    xaxis=dict(showbackground=False, showgrid=False, zeroline=False,
               showticklabels=False, title=''),
    yaxis=dict(showbackground=False, showgrid=False, zeroline=False,
               showticklabels=False, title=''),
    zaxis=dict(showbackground=False, showgrid=False, zeroline=False,
               showticklabels=False, title=''),
    # optional: aspect ratio
    aspectmode="data"    
)

# create subplot figure with 3D subplots
fig = make_subplots(
    rows=rows, cols=cols,
    specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
    subplot_titles=[f"{all_jsdiv[u]:.4f}" for u in jsidx],
    horizontal_spacing = 0 ,  # reduce horizontal gap (0 = no gap)
    vertical_spacing = 0     # reduce vertical gap
)

# loop and add traces
for i, u in enumerate(jsidx):
    r, c = divmod(i, cols)

    # call your existing function
    best_js = np.array([
        data_10[u]["true_value"],
        data_10[u]["predictions"].numpy()
    ])
    subfig = viz.compare_plot(best_js)  # <- returns a plotly figure

    # extract traces and add them into subplot
    for trace in subfig.data:
        fig.add_trace(trace, row=r+1, col=c+1)
        
# Apply to all subplot scenes
layout_updates = {}
for i in range(1, rows*cols + 1):
    scene_name = "scene" if i == 1 else f"scene{i}"
    layout_updates[scene_name] = clean_scene

fig.update_layout(
    margin=dict(l=0, r=0, t=30, b=0),  # trim margins
    showlegend = False,
    **layout_updates
)

fig.write_image("SKL_best.png", width=1250, height=500)
fig.show(renderer="browser")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_weight_space_pca(results_list):
    """
    Performs PCA on the readout weights to visualize the solution space.
    
    Args:
        results_list: The list returned by parameter_sweep()
                      Each item must have 'readout_weights', 'metrics', and 'parameters'.
    """
    
    # 1. Prepare Data Containers
    weight_vectors = []
    mses = []
    jsdivs = []
    param_info = [] # Store text info for the best model

    print(f"Processing {len(results_list)} results...")

    for i, res in enumerate(results_list):
        # Extract weights
        # Shape is usually (Output_Dim, Reservoir_Size)
        w_matrix = res['readout_weights'] 
        
        # Extract Metric (RMSE)
        # Handle cases where RMSE might be inside a dict or just a float
        try:
            mse_val = res['metrics']['RMSE']
            jsdiv = chaos_utils.js_divergence(res["true_value"], res["predictions"])
        except KeyError:
            print(f"Skipping index {i}: RMSE not found")
            continue

        # --- CRITICAL SAFETY CHECK ---
        # Reservoir weights often explode to Infinity or NaN with bad parameters.
        # PCA will crash if we don't filter these out.
        if np.isnan(w_matrix).any() or np.isinf(w_matrix).any() or np.isnan(mse_val) or np.isinf(mse_val):
            # print(f"Skipping index {i}: unstable solution (NaN/Inf).")
            continue

        # Flatten the matrix into a 1D vector so PCA can eat it
        # e.g., shape (3, 100) -> (300,)
        w_flat = w_matrix.flatten()
        
        weight_vectors.append(w_flat)
        mses.append(mse_val)
        jsdivs.append(jsdiv)
        
        # Store parameter string for labeling
        p = res['parameters']
        p_str = f"SR:{p['SpectralRadius']:.2f}, LR:{p['LeakyRate']:.2f}, IS:{p['InputScaling']:.2f}"
        param_info.append(p_str)

    # Convert to NumPy array
    X = np.array(weight_vectors)
    mses = np.array(mses)
    jsdivs = np.array(jsdivs)

    if len(X) < 2:
        print("Not enough valid models to perform PCA.")
        return

    # 2. Run PCA
    # We reduce the high-dimensional weight space to 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    # 3. Plotting
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    # cmap='viridis_r' reverses colors so Dark Purple = Low Error (Good), Yellow = High Error (Bad)
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=jsdivs, cmap='viridis_r', s=80, alpha=0.8, edgecolors='k', linewidth=0.5)
    
    # Colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('JSDiv (lighter is Better)')

    # Titles and Labels
    plt.title(f"PCA of Reservoir Readout Weights ({len(X)} valid runs)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

# --- CONFIGURATION ---
# Point this to the .pkl file generated by main_sweep.py
# ---------------------

def load_and_parse(path):
    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        results = pickle.load(f)
    
    data = []
    for res in results:
        # Flatten dictionary for DataFrame
        row = {
            'SR': res['parameters']['SpectralRadius'],
            'LR': res['parameters']['LeakyRate'],
            'IS': res['parameters']['InputScaling'],
            'RMSE': res['metrics']['RMSE']
        }
        data.append(row)
    
    return pd.DataFrame(data)

def plot_analysis(df):
    # Filter out exploded runs (NaN or Infinity)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Plotting {len(df)} valid runs...")

    # Set up the figure
    fig = plt.figure(figsize=(18, 10))
    
    # --- PLOT 1: 3D Parameter Space ---
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Color by RMSE (reversed so dark/purple is bad, bright/yellow is good usually, 
    # but let's use 'viridis_r' where Dark = Good (Low RMSE), Yellow = Bad)
    img = ax1.scatter(df['SR'], df['LR'], df['IS'], c=df['RMSE'], cmap='viridis_r', alpha=0.8, edgecolors='k', linewidth=0.3)
    
    ax1.set_xlabel('Spectral Radius')
    ax1.set_ylabel('Leaky Rate')
    #ax1.set_zlabel('Input Scaling') #Type:ignore
    ax1.set_title('3D Parameter Landscape (Color = RMSE)')
    fig.colorbar(img, ax=ax1, label='RMSE (Darker is Better)')

    # --- PLOT 2: Parallel Coordinates (The Flow) ---
    # This helps visualize "To get low RMSE, I need High SR AND Low LR" relationships
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Normalize data for parallel coordinates visualization
    df_norm = (df - df.min()) / (df.max() - df.min())
    # Add back raw RMSE for coloring
    df_norm['RMSE_RAW'] = df['RMSE']
    
    # Sort so best models are drawn last (on top)
    df_norm = df_norm.sort_values('RMSE_RAW', ascending=False)
    
    for i, row in df_norm.iterrows():
        # Color based on performance
        color = plt.cm.viridis_r((row['RMSE_RAW'] - df['RMSE'].min()) / (df['RMSE'].max() - df['RMSE'].min())) #type:ignore
        alpha = 0.8 if row['RMSE_RAW'] < df['RMSE'].quantile(0.1) else 0.1 # Highlight top 10%
        
        ax2.plot(['SR', 'LR', 'IS'], [row['SR'], row['LR'], row['IS']], color=color, alpha=alpha)
        
    ax2.set_title('Parallel Coordinates (Top 10% Highlighted)')
    ax2.set_ylabel('Normalized Parameter Value (0-1)')
    ax2.grid(True, alpha=0.3)

    # --- PLOT 3: Pairwise Correlations ---
    ax3 = fig.add_subplot(2, 2, 3)
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3, vmin=-1, vmax=1)
    ax3.set_title('Correlation Matrix')

    # --- PLOT 4: Best vs Worst Distributions ---
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Split into Top 20% vs Bottom 20%
    threshold = df['RMSE'].quantile(0.2)
    top_models = df[df['RMSE'] < threshold]
    
    ax4.hist(top_models['SR'], alpha=0.5, label='Best SR', bins=10, density=True)
    ax4.hist(top_models['LR'], alpha=0.5, label='Best LR', bins=10, density=True)
    ax4.hist(top_models['IS'], alpha=0.5, label='Best IS', bins=10, density=True)
    ax4.legend()
    ax4.set_title('Parameter Distribution of Top 20% Models')
    ax4.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # --- Print Text Summary ---
    print("\n--- TOP 5 CONFIGURATIONS ---")
    print(df.sort_values('RMSE').head(5))

if __name__ == "__main__":
    # Check if file exists
    import os
    if not os.path.exists(file):
        print(f"Error: Could not find results file at: {file}")
        print("Please run main_sweep.py first or update RESULT_PATH in this script.")
    else:
        df = load_and_parse(file)
        #plot_analysis(df)
        viz.plot_parallel_coordinates(df)
        

# --- Example Usage ---
# Assuming 'results' is the list returned from your parameter_sweep function
        plot_weight_space_pca(data_10)