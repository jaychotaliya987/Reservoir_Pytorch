import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from dysts.flows import Lorenz

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from reservoirgrid.datasets import LorenzAttractor
from reservoirgrid.models import Reservoir
from reservoirgrid.helpers import utils

import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("Imports Done!\n")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

def from_dysts(length = 1000, newgen = False):
    if not newgen:
        Dataset = np.genfromtxt("reservoirgrid/datasets/Lorenz_fine.csv", delimiter=",", skip_header=1)[:,1:]
        Dataset = (Dataset - Dataset.min()) / (Dataset.max() - Dataset.min()) *2 -1
        inputs, targets = torch.tensor(Dataset[:-1], dtype = torch.float32), torch.tensor(Dataset[1:], dtype = torch.float32)
        train_inputs, test_inputs = train_test_split(inputs, shuffle=False, test_size=0.2, random_state=42)
        train_targets, test_targets = train_test_split(targets, shuffle=False, test_size=0.2, random_state=42)
    else:
        Dataset = Lorenz().make_trajectory(length)
        inputs, targets = torch.tensor(Dataset[:-1], dtype = torch.float32), torch.tensor(Dataset[1:], dtype = torch.float32)
        train_inputs, test_inputs = train_test_split(inputs, shuffle=False, test_size=0.2, random_state=42)
        train_targets, test_targets = train_test_split(targets, shuffle=False, test_size=0.2, random_state=42)
    return train_inputs, train_targets, test_inputs, test_targets

def from_mygen():
# Generate the Lorenz Attractor data
    attractor = LorenzAttractor(sample_len=10000, n_samples=1, xyz=[1.0, 1.0, 1.0], 
                            sigma=10.0, b=8/3, r=28.0, seed=42)
    attractor_samp = attractor[0]

    # Normalization is very important
    attractor_samp = (attractor_samp - attractor_samp.min()) / (attractor_samp.max() - attractor_samp.min())
    inputs ,targets = attractor_samp[:-1], attractor_samp[1:]
    train_inputs, test_inputs = train_test_split(inputs, test_size=0.2, shuffle=False, random_state=42)
    train_targets, test_targets = train_test_split(targets, test_size=0.2, shuffle=False, random_state=42)
    return train_inputs, train_targets, test_inputs, test_targets

train_inputs, train_targets, test_inputs, test_targets = from_mygen()


ResLorenz = Reservoir(
    input_dim=3,
    reservoir_dim=1300,
    output_dim=3,
    spectral_radius=1,
    leak_rate=0.5,
    sparsity=0.9,
    input_scaling=0.5,
    noise_level = 0.01
)

ResLorenz.train_readout(train_inputs, train_targets, warmup=1000)
time_steps = np.arange(len(test_targets))

# Generate predictions using test inputs
with torch.no_grad():
    predictions = ResLorenz.predict(train_inputs, steps=len(test_targets))

error = utils.RMSE(test_targets[:],predictions[:])
print(f"RMSE: {error:.4f}")
predictions = predictions.cpu().numpy()
test_targets_np = test_targets.cpu().numpy()

# Enhanced color scheme matching original
colors = {
    'train': '#8da0cb',      
    'test': '#66c2a5',       
    'prediction': '#fc8d62', 
    'divider': 'gray',       
    'background': 'rgb(240, 240, 240)',
    'grid': 'rgb(200, 200, 200)'
}

# Prepare training data for plotting (last 1000 points)
train_targets_np = train_targets.cpu().numpy()[-1000:]  # Last portion of training data
train_time_steps = np.arange(-len(train_targets_np), 0) # Negative time steps for training

fig_components = make_subplots(
    rows=3, cols=1, 
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=("X Component", "Y Component", "Z Component")
)

# Plot each component
for i, component in enumerate(['X', 'Y', 'Z'], 1):
    # Training data (before prediction start)
    fig_components.add_trace(go.Scatter(
        x=train_time_steps,
        y=train_targets_np[:, i-1],
        mode='lines',
        line=dict(color=colors['train'], width=2.5),
        name='Training Data',
        legendgroup='train',
        showlegend=True if i==1 else False
    ), row=i, col=1)
    
    # Test data (after prediction start)
    fig_components.add_trace(go.Scatter(
        x=time_steps,
        y=test_targets_np[:, i-1],
        mode='lines',
        line=dict(color=colors['test'], width=2.5),
        name='Test Data',
        legendgroup='test',
        showlegend=True if i==1 else False
    ), row=i, col=1)
    
    # Predictions (after prediction start)
    fig_components.add_trace(go.Scatter(
        x=time_steps,
        y=predictions[:, i-1],
        mode='lines',
        line=dict(color=colors['prediction'], width=2.5),
        name='Prediction',
        legendgroup='prediction',
        showlegend=True if i==1 else False
    ), row=i, col=1)
    
    # Prediction start line
    fig_components.add_vline(
        x=0, 
        line=dict(color=colors['divider'], width=1, dash='dot'),
        row=i, col=1
    )

# Add annotation for prediction start
fig_components.add_annotation(
    x=0,
    y=np.max(test_targets_np[:, 0]),
    text="Prediction Start",
    showarrow=True,
    arrowhead=1,
    ax=-60,
    ay=-30,
    row=1,
    col=1
)

# Enhanced layout matching original style
fig_components.update_layout(
    title=dict(
        text="<b>Lorenz Attractor: Training, Test and Predictions</b>",
        y=0.95,
        x=0.5,
        font=dict(size=24, family='Arial')
    ),
    height=800,
    width=1000,
    template='plotly_white',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(t=100),
    hovermode='x unified',
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Axis labels
for i in range(1,4):
    fig_components.update_yaxes(
        title_text="Value", 
        row=i, col=1,
        gridcolor='rgba(200,200,200,0.3)'
    )

# X-axis settings to show training and test periods
fig_components.update_xaxes(
    title_text="Time Step (Prediction starts at 0)", 
    row=3, col=1,
    gridcolor='rgba(200,200,200,0.3)',
    range=[-200, len(test_targets)]  # Show some training period and all test
)

fig_components.show()

# Enhanced color scheme matching component plots
colors = {
    'train': '#8da0cb',      # Light purple-blue
    'test': '#66c2a5',       # Teal
    'prediction': '#fc8d62', # Orange
    'divider': 'gray',
    'background': 'rgb(240, 240, 240)',
    'grid': 'rgb(200, 200, 200)'
}

# Create 3D comparison plot with matching colors
fig_compare = go.Figure()

# Add test attractor (semi-transparent teal)
fig_compare.add_trace(go.Scatter3d(
    x=test_targets_np[:, 0],
    y=test_targets_np[:, 1],
    z=test_targets_np[:, 2],
    mode='lines',
    line=dict(
        color=colors['test'],
        width=4,
        
    ),
    opacity=0.7,  # Semi-transparent
    name='Test Attractor',
    hoverinfo='none'
))

# Add predictions (solid orange)
fig_compare.add_trace(go.Scatter3d(
    x=predictions[:, 0],
    y=predictions[:, 1],
    z=predictions[:, 2],
    mode='lines',
    line=dict(
        color=colors['prediction'],
        width=4
    ),
    name='Predicted Attractor',
    hoverinfo='none'
))

# Enhanced layout with matching style
fig_compare.update_layout(
    title={
        'text': "<b>Lorenz Attractor: Test vs Prediction</b>",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=24, family='Arial')
    },
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis',
        xaxis=dict(
            gridcolor=colors['grid'], 
            showbackground=True,
            backgroundcolor=colors['background']
        ),
        yaxis=dict(
            gridcolor=colors['grid'],
            showbackground=True,
            backgroundcolor=colors['background']
        ),
        zaxis=dict(
            gridcolor=colors['grid'],
            showbackground=True,
            backgroundcolor=colors['background']
        ),
        bgcolor=colors['background'],
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.6) 
        )
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    paper_bgcolor='white',
    height=700,
    width=800,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        font=dict(size=12),
    #font=dict(family="Arial", size=12)
))

# Add a subtle colorbar to show Z dimension
fig_compare.update_traces(
    marker=dict(
        showscale=True,
        colorscale='Viridis',  # Keep Z-dimension coloring
        colorbar=dict(
            thickness=20,
            x=0.9,
            len=0.5,
            title='Z value'
        )
    ),
    selector={'type':'scatter3d'}
)

fig_compare.show()