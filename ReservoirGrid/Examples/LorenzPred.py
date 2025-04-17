import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from Datasets.LorenzAttractor import LorenzAttractor
from Models.Reservoir import Reservoir

import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("Imports Done!\n")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU is available')
else:
    device = torch.device('cpu')
    print('CPU is available')

# Generate the Lorenz Attractor data
attractor = LorenzAttractor(sample_len=10000, n_samples=1, xyz=[1.0, 1.0, 1.0], 
                            sigma=10.0, b=8/3, r=28.0, seed=42)

attractor_samp = attractor[0]

attractor_samp = (attractor_samp - attractor_samp.min()) / (attractor_samp.max() - attractor_samp.min())

inputs = attractor_samp[:-1,:].to(device)
targets = attractor_samp[1:,:].to(device)

print("Input shape:", inputs.shape)
print("Target shape:", targets.shape)


ResLorenz = Reservoir(
    input_dim=3, 
    reservoir_dim=1000, 
    output_dim=3, 
    spectral_radius=0.95, 
    leak_rate=0.3, 
    sparsity=0.9, 
    input_scaling=0.5
)

ResLorenz.to(device)


ResLorenz.train_readout(inputs, targets, Warmup=200)
predictions = ResLorenz.predict(inputs, steps=10000)
predictions = predictions.cpu().detach()

fig = go.Figure(data=[go.Scatter3d(
    x=predictions[:,0].numpy(), 
    y=predictions[:,1].numpy(), 
    z=predictions[:,2].numpy(),
    mode='lines',
    line=dict(
        color=predictions[:,2].numpy(),  
        colorscale='Viridis',            
        width=3,                         
        showscale=True                   
    ),
    hoverinfo='none'                   
)])

fig.update_layout(
    title={
        'text': "<b>Lorenz Attractor Predictions</b>",
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
        xaxis=dict(gridcolor='rgb(200, 200, 200)', showbackground=True),
        yaxis=dict(gridcolor='rgb(200, 200, 200)', showbackground=True),
        zaxis=dict(gridcolor='rgb(200, 200, 200)', showbackground=True),
        bgcolor='rgb(240, 240, 240)',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.6) 
        )
    ),
    margin=dict(l=0, r=0, b=0, t=30),  
    paper_bgcolor='white',
    height=700,                        
    width=800
)


fig.show()


# Create figure with subplots
fig = make_subplots(rows=3, cols=1, 
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=("X Component", "Y Component", "Z Component"))

# Color scheme
colors = ['#8da0cb', '#66c2a5', '#fc8d62']
divider_color = 'gray'

# Plot each component
components = ['X', 'Y', 'Z']
for i, component in enumerate(components, 1):
    fig.add_trace(go.Scatter(
        x=np.arange(len(attractor_samp[-1000:, i-1])),
        y=attractor_samp[-1000:, i-1].numpy(),
        mode='lines',
        line=dict(color=colors[0], width=2.5),
        name='Target',
        legendgroup='target',
        showlegend=True if i==1 else False
    ), row=i, col=1)
    
    # Predictions
    prediction_start = len(attractor_samp[-1000:, i-1]) - 1
    fig.add_trace(go.Scatter(
        x=np.arange(prediction_start, prediction_start + len(predictions)),
        y=predictions[:, i-1].numpy(),
        mode='lines',
        line=dict(color=colors[2], width=2.5, dash='solid'),
        name='Prediction',
        legendgroup='prediction',
        showlegend=True if i==1 else False
    ), row=i, col=1)
    
    # Prediction start line
    fig.add_vline(
        x=prediction_start, 
        line=dict(color=divider_color, width=1, dash='dot'),
        row=i, col=1
    )
    
    # Annotation for first component
    if i == 1:
        fig.add_annotation(
            x=prediction_start,
            y=np.max(attractor_samp[-1000:, i-1].numpy()),
            text="Prediction Start",
            showarrow=True,
            arrowhead=1,
            ax=-60,
            ay=-30,
            row=1,
            col=1
        )

# Update layout
fig.update_layout(
    title=dict(
        text="<b>Lorenz Attractor: Components and Predictions</b>",
        y=0.95,
        x=0.5,
        font=dict(size=24)
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
    hovermode='x unified'
)

for i in range(1,4):
    fig.update_yaxes(title_text="Value", row=i, col=1)

# X-axis for bottom plot
fig.update_xaxes(title_text="Time Step", row=3, col=1)

# Style
fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
    font=dict(family="Arial", size=12)
)

fig.show()