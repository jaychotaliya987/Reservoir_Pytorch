import plotly.graph_objects as go
import numpy as np
import torch

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from reservoirgrid.datasets import LorenzAttractor


attractor = LorenzAttractor(sample_len=10000, n_samples=1, xyz=[1.0, 1.0, 1.0], 
                            sigma=10.0, b=8/3, r=28.0, seed=42)
attractor_samp = attractor[0].numpy()  # Convert to numpy array

colors = np.linspace(0, 1, len(attractor_samp))

fig = go.Figure()

import plotly.graph_objects as go
import numpy as np

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=attractor_samp[:, 0],
    y=attractor_samp[:, 1],
    z=attractor_samp[:, 2],
    mode='lines',
    line=dict(
        width=3,  
        color=attractor_samp[:, 2],  
        colorscale='Viridis',  
        showscale=True,
        colorbar=dict(
            title='Z-value',
            thickness=20,
            len=0.75,
            yanchor='middle',
            y=0.5
        )
    ),
    name='Trajectory',
    hoverinfo='none'
))

# Start and end markers
fig.add_trace(go.Scatter3d(
    x=[attractor_samp[0, 0]],
    y=[attractor_samp[0, 1]],
    z=[attractor_samp[0, 2]],
    mode='markers',
    marker=dict(size=5, color='limegreen'), 
    name='Start',
    hoverinfo='name'
))

fig.add_trace(go.Scatter3d(
    x=[attractor_samp[-1, 0]],
    y=[attractor_samp[-1, 1]],
    z=[attractor_samp[-1, 2]],
    mode='markers',
    marker=dict(size=5, color='crimson'),
    name='End',
    hoverinfo='name'
))

fig.update_layout(
    title={
        'text': "<b>Lorenz Attractor</b>",
        'y':0.92,
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
    width=900,
    height=700,
    margin=dict(l=0, r=0, b=0, t=30),
    paper_bgcolor='white',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor='rgba(255,255,255,0.7)'
    )
)

fig.show()