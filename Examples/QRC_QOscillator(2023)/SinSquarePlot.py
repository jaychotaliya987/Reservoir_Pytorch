import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from reservoirgrid.datasets import SineSquare

print("-------------------------")
print("Imports Done!")
print("-------------------------")

sample_len = 100
dataset = SineSquare(sample_len)
data, label = dataset.get_all()
data = data.flatten() # removes the batching and get the data into a single row vector for ploting


if data.ndim != 1: raise "Input is not 1D"

fig = go.Figure()

# Add the line trace
fig.add_trace(go.Scatter(
    y=data,
    x=np.arange(0, 2*sample_len, 2*np.pi/100),
    mode='lines+markers',  # This adds both line and points
    line=dict(
        width=3,
        color='#5DADE2',  # Nice blue color
    ),
    marker=dict(
        size=8,
        color='#F39C12',   # Orange color for markers
        line=dict(
            width=2,
            color='#FFFFFF'  # White border for markers
        ),
        symbol='circle' # Fancy marker shape
    ),
    name='Signal',
    hovertemplate='Time: %{x:.2f}<br>Value: %{y:.2f}<extra></extra>'
))
fig.update_layout(
    title='Sine-Square Wave Visualization',
    title_font=dict(size=24, family='Arial', color='black'),  # Changed to black
    xaxis_title='Time (radians)',
    yaxis_title='Amplitude',
    plot_bgcolor='white',  # White background
    paper_bgcolor='white',
    font=dict(color='black', family='Arial'),  # Black text
    xaxis=dict(
        gridcolor='rgba(0, 0, 0, 0.1)',  # Light gray grid
        linecolor='rgba(0, 0, 0, 0.5)',  # Darker axis line
        showgrid=True
    ),
    yaxis=dict(
        gridcolor='rgba(0, 0, 0, 0.1)',  # Light gray grid
        linecolor='rgba(0, 0, 0, 0.5)',  # Darker axis line
        showgrid=True
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color='black')  # Black legend text
    ),
    margin=dict(l=50, r=50, b=50, t=80),
    hoverlabel=dict(
        bgcolor='rgba(255,255,255,0.9)',  # White hover background
        font_size=14,
        font_family="Arial",
        bordercolor='rgba(0,0,0,0.2)',  # Light border
        font_color='black'  # Black hover text
    )
)

fig.show()
