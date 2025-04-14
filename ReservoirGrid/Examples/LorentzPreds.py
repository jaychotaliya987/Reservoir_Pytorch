import plotly.graph_objects as go
import numpy as np
import torch
from Datasets.LorenzAttractor import LorenzAttractor

# Generate the Lorenz Attractor data
attractor = LorenzAttractor(sample_len=10000, n_samples=10, xyz=[1.0, 1.0, 1.0], sigma=10.0, b=8/3, r=28.0)
attractor_samp = attractor[1].numpy()  # Convert to numpy array

# Create time-based color gradient
colors = np.linspace(0, 1, len(attractor_samp))

# Create the figure
fig = go.Figure()

# Add the main trajectory with color gradient
fig.add_trace(go.Scatter3d(
    x=attractor_samp[:, 0],
    y=attractor_samp[:, 1],
    z=attractor_samp[:, 2],
    mode='lines',
    line=dict(
        width=4,
        color=colors,
        colorscale='Plasma',
        showscale=True,
        colorbar=dict(
            title='Time Progression',
            thickness=20,
            len=0.75,
            yanchor='middle',
            y=0.5
        )
    ),
    name='Trajectory'
))

# Add start and end markers
fig.add_trace(go.Scatter3d(
    x=[attractor_samp[0, 0]],
    y=[attractor_samp[0, 1]],
    z=[attractor_samp[0, 2]],
    mode='markers',
    marker=dict(size=6, color='green'),
    name='Start Point'
))

fig.add_trace(go.Scatter3d(
    x=[attractor_samp[-1, 0]],
    y=[attractor_samp[-1, 1]],
    z=[attractor_samp[-1, 2]],
    mode='markers',
    marker=dict(size=6, color='red'),
    name='End Point'
))

# Customize the layout
fig.update_layout(
    title={
        'text': "<b>Lorenz Attractor Trajectory</b>",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=24)
    },
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis',
        xaxis=dict(
            backgroundcolor='rgba(240,240,240,0.5)',
            gridcolor='white',
            showbackground=True
        ),
        yaxis=dict(
            backgroundcolor='rgba(240,240,240,0.5)',
            gridcolor='white',
            showbackground=True
        ),
        zaxis=dict(
            backgroundcolor='rgba(240,240,240,0.5)',
            gridcolor='white',
            showbackground=True
        ),
        camera=dict(
            eye=dict(x=1.25, y=1.25, z=0.75)  # Initial view angle
        )
    ),
    width=1000,
    height=800,
    margin=dict(l=0, r=0, b=0, t=90),
    paper_bgcolor='rgba(245,245,245,1)',
    scene_camera=dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
    ),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# Add rotation animation button
fig.update_layout(
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(
            label="Play Rotation",
            method="animate",
            args=[None, dict(frame=dict(duration=50, redraw=True), 
                            fromcurrent=True)]
        )],
        pad=dict(r=10, t=87),
        showactive=False,
        x=0.1,
        xanchor="right",
        y=0,
        yanchor="top"
    )]
)

# Create rotation frames for animation
frames = []
for angle in range(0, 360, 5):
    frames.append(go.Frame(
        layout=dict(
            scene_camera=dict(eye=dict(
                x=1.25 * np.cos(np.radians(angle)),
                y=1.25 * np.sin(np.radians(angle)),
                z=0.75
            )
        )
    )))

fig.frames = frames

fig.show()
fig.write_html("LorenzAttractor3D.html")