import plotly.graph_objects as go
import numpy as np
import torch
from Datasets.LorenzAttractor import LorenzAttractor

# Generate the Lorenz Attractor data
attractor = LorenzAttractor(sample_len=10000, n_samples=10, xyz=[1.0, 1.0, 1.0], sigma=10.0, b=8/3, r=28.0, seed=42)
attractor_samp = attractor[1].numpy()  # Convert to numpy array

# Create time-based color gradient
colors = np.linspace(0, 1, len(attractor_samp))

# Create the figure
fig = go.Figure()

import plotly.graph_objects as go
import numpy as np

# Create the figure with consistent styling
fig = go.Figure()

# Main trajectory with improved color gradient (consistent with first plot)
fig.add_trace(go.Scatter3d(
    x=attractor_samp[:, 0],
    y=attractor_samp[:, 1],
    z=attractor_samp[:, 2],
    mode='lines',
    line=dict(
        width=3,  # Slightly thinner than your second version
        color=attractor_samp[:, 2],  # Color by z-value like first plot
        colorscale='Viridis',  # Consistent with first plot
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

# Start and end markers (from second plot but more subtle)
fig.add_trace(go.Scatter3d(
    x=[attractor_samp[0, 0]],
    y=[attractor_samp[0, 1]],
    z=[attractor_samp[0, 2]],
    mode='markers',
    marker=dict(size=5, color='limegreen'),  # More subtle than bright green
    name='Start',
    hoverinfo='name'
))

fig.add_trace(go.Scatter3d(
    x=[attractor_samp[-1, 0]],
    y=[attractor_samp[-1, 1]],
    z=[attractor_samp[-1, 2]],
    mode='markers',
    marker=dict(size=5, color='crimson'),  # More subtle than bright red
    name='End',
    hoverinfo='name'
))

# Layout consistent with first plot but enhanced
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
        xaxis=dict(
            gridcolor='rgb(200, 200, 200)',
            showbackground=True,
            backgroundcolor='rgb(240, 240, 240)'
        ),
        yaxis=dict(
            gridcolor='rgb(200, 200, 200)',
            showbackground=True,
            backgroundcolor='rgb(240, 240, 240)'
        ),
        zaxis=dict(
            gridcolor='rgb(200, 200, 200)',
            showbackground=True,
            backgroundcolor='rgb(240, 240, 240)'
        ),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.6)  # Consistent with first plot
        )
    ),
    width=900,
    height=700,
    margin=dict(l=0, r=0, b=0, t=80),
    paper_bgcolor='white',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor='rgba(255,255,255,0.7)'
    )
)

# Optional: Add subtle animation (less prominent than second version)
fig.update_layout(
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(
            label="▶️ Rotate",
            method="animate",
            args=[None, dict(frame=dict(duration=50, redraw=True), 
                            fromcurrent=True,
                            mode='immediate')],
        )],
        pad=dict(r=10, t=10),
        showactive=True,
        x=0.05,
        xanchor="left",
        y=-0.1,
        yanchor="top"
    )]
)

# Create smooth rotation frames
frames = []
for angle in range(0, 360, 2):
    frames.append(go.Frame(
        layout=dict(
            scene_camera=dict(eye=dict(
                x=1.5 * np.cos(np.radians(angle)),
                y=1.5 * np.sin(np.radians(angle)),
                z=0.6
            ))
    )))

fig.frames = frames

fig.show()