import numpy as np
from matplotlib.colors import to_hex
import matplotlib.cm as cm
import torch
from typing import Union

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.express import histogram

from ..models import Reservoir

def compare_plot(datasets, title=None, legend_names=None ,figsize=(1080, 600), colorscale='Viridis', 
                 line_width=3, marker_size=2, bgcolor='rgb(240, 240, 240)', **kwargs) -> go.Figure:
    """
    Create plot of datasets with trajectories overlaid. Useful to compare true vs predicted.
    can dynamically handle 2D and 3D data.
    
    Args:
        datasets (list): List of numpy arrays. If multiple, combine them in a array. ex. [dataset1, dataset2, ...]
        titles (list): Optional list of titles for each dataset
        figsize (tuple): Figure size (width, height)
        colorscale: Plotly colorscale name (for 3D data)
        line_width: Width of plot lines
        marker_size: Size of start/end markers
        bgcolor: Background color

    Returns: 
        plotly.graph_objects.Figure

    Note: Uses Abstractions

    """

    n = len(datasets)
    if legend_names is None:
        legend_names = [f'Dataset {i+1}' for i in range(n)]
    
    # Color sequence for different trajectories
    colors = px.colors.qualitative.Plotly
    
    dim = datasets[0].shape[1] if datasets[0].ndim > 1 else 1
    
    fig = go.Figure()

    line_trace, marker_trace = _get_trace_builders(dim)

    for i, data in enumerate(datasets):
        color = colors[i % len(colors)]

        if dim == 1:
            x = np.arange(len(data))
            y = data

            fig.add_trace(trace = line_trace(x, y, name = legend_names[i], color = color, lw = line_width))     # type: ignore
            fig.add_trace(trace = marker_trace(x[0], y[0], name = "", color = "limegreen", size = marker_size)) # type: ignore
            fig.add_trace(trace = marker_trace(x[-1], y[-1], name = "", color = "crimson", size = marker_size)) # type: ignore

        elif dim == 2:
            x, y = data[:,0], data[:,1]

            fig.add_trace(trace = line_trace(x, y, name = legend_names[i], color = color, lw = line_width))      # type: ignore
            fig.add_trace(trace = marker_trace(x[0], y[0], name = "", color = "limegreen", size = marker_size)) # type: ignore
            fig.add_trace(trace = marker_trace(x[-1], y[-1], name = "", color = "crimson", size = marker_size)) # type: ignore

        else:
            x, y, z = data[:,0], data[:,1], data[:,2]

            fig.add_trace(trace = line_trace(x, y, z, legend_names[i], color, line_width))          # type: ignore
            fig.add_trace(trace = marker_trace(x[0], y[0], z[0], "", "limegreen",   marker_size))   # type: ignore
            fig.add_trace(trace = marker_trace(x[-1], y[-1], z[-1], "", "crimson",  marker_size))   # type: ignore

    return fig

def plot_components(trajectory, time=None, labels=None, title=None, 
                   figsize=(1080,600), colorscale='Viridis', line_width=2.5,
                   bgcolor='rgb(240, 240, 240)', title_fontsize=20) -> go.Figure:
    """
    Create beautiful interactive component plot using Plotly.
    
    Args:
        trajectory (np.ndarray): Shape (n_points, n_components)
        time (np.ndarray): Custom time values (default: indices)
        labels (list): Component names (e.g., ['x', 'y', 'z'])
        title (str): Overall title
        figsize (tuple): Figure size (width, height)
        colorscale: Plotly colorscale name
        line_width: Width of plot lines
        bgcolor: Background color
        title_fontsize: Font size for title
    """
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    n_components = trajectory.shape[1]
    time = np.arange(trajectory.shape[0]) if time is None else time
    
    if labels is None:
        labels = [f'Component {i+1}' for i in range(n_components)]
    
    # Create subplot figure
    fig = make_subplots(
        rows=n_components, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05, 
        subplot_titles=labels,
        specs=[[{'type': 'xy'}] for _ in range(n_components)]  # Proper 2D specs structure
    )
    
    # Get colors from colorscale
    colors = [to_hex(cm.get_cmap('viridis') (i/n_components)) for i in range(n_components)]
    
    for i in range(n_components):
        fig.add_trace(go.Scatter(
            x=time,
            y=trajectory[:,i],
            mode='lines',
            line=dict(width=line_width, color=colors[i]),
            name=labels[i],
            hoverinfo='x+y',
            showlegend=False
        ), row=i+1, col=1)
        
        # Customize each subplot
        fig.update_yaxes(title_text=labels[i], row=i+1, col=1)
        fig.update_xaxes(showgrid=True, gridcolor='rgb(200, 200, 200)', row=i+1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='rgb(200, 200, 200)', row=i+1, col=1)
    
    # Update layout
    fig.update_layout(
        height=figsize[1],
        width=figsize[0],
        title={
            'text': f"<b>{title}</b>" if title else "",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=title_fontsize, family='Arial')
        },
        margin=dict(l=50, r=50, b=50, t=80 if title else 50),
        paper_bgcolor='white',
        plot_bgcolor=bgcolor,
        hovermode='x unified',
        font=dict(family='Arial', size=12))
    
    # Only show x-axis label on bottom plot
    fig.update_xaxes(title_text='Time', col=1)   
    return fig


def visualize_reservoir_states(
    model: Union[Reservoir, dict], 
    n_units: int = 8,
    time_window: tuple = (None, None),
    plot_type: str = 'line',
    show_distribution: bool = True
):
    """
    Enhanced visualization of reservoir unit activations with multiple analysis tools.
    
    Args:
        model: Reservoir model or result dictionary containing 'reservoir_states'
        n_units: Number of reservoir units to visualize (randomly sampled)
        time_window: Tuple of (start, end) indices to zoom in on specific time period
        plot_type: Type of visualization ('line', 'heatmap', or 'both')
        show_distribution: Whether to show activation distribution histograms
    
    Returns:
        plotly.graph_objects.Figure: Interactive visualization figure
    
    Interpretation Guide:
        - Healthy reservoirs show:
            * Diverse activation patterns (not all identical)
            * Reasonable amplitude (-1 to 1 range)
            * Mix of periodic and chaotic behaviors
        - Problem signs:
            * Flat lines → Dead units
            * Saturated activations → Poor input scaling
            * Synchronized units → Lack of separation
            * Exploding values → Stability issues
    """
    # Extract states from model or results dict
    if isinstance(model, dict):
        states = model.get('reservoir_states', None)
        if states is None:
            raise ValueError("Result dictionary must contain 'reservoir_states'")
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
    else:
        if not hasattr(model, 'reservoir_states'):
            raise AttributeError("Model must have 'reservoir_states' attribute")
        states = model.reservoir_states.cpu().numpy()
    
    # Transpose to (time_steps, units) if needed
    if states.shape[0] > states.shape[1]:
        states = np.squeeze(states)
    

    # Apply time window
    start, end = time_window
    states = states[:, start:end] if None not in time_window else states
    
    # Randomly sample units if needed
    if n_units < states.shape[0]:
        unit_indices = np.random.choice(
            states.shape[0], 
            size=min(n_units, states.shape[0]), 
            replace=False
        )
        states = states[unit_indices, :]
    
    # Create figure
    fig = go.Figure()
    time_steps = np.arange(states.shape[1])
    
    if plot_type in ['line', 'both']:
        for i, unit_states in enumerate(states):
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=unit_states,
                name=f'Unit {i}',
                mode='lines',
                opacity=0.7,
                line=dict(width=1),
                hoverinfo='x+y+name'
            ))
    
    if plot_type in ['heatmap', 'both']:
        heatmap_fig = px.imshow(
            states,
            aspect='auto',
            labels=dict(x="Time Step", y="Unit", color="Activation"),
            color_continuous_scale='RdBu'
        )
        if plot_type == 'heatmap':
            return heatmap_fig
        
    if show_distribution:
        hist_fig = histogram(
            states.T,
            nbins=50,
            labels={'value': 'Activation'},
            title='Activation Distribution',
            opacity=0.7,
            marginal='box'
        )
    
    # Layout configuration
    fig.update_layout(
        title=f'Reservoir Unit Dynamics ({states.shape[0]} units shown)',
        xaxis_title='Time Step',
        yaxis_title='Activation',
        hovermode='x unified',
        height=600,
        template='plotly_white'
    )
    
    if show_distribution:
        from plotly.subplots import make_subplots
        combined_fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Temporal Dynamics', 'Activation Distribution'),
            vertical_spacing=0.1
        )
        
        for trace in fig.data:
            combined_fig.add_trace(trace, row=1, col=1)
        
        for trace in hist_fig.data: # type: ignore
            combined_fig.add_trace(trace, row=2, col=1)
        
        combined_fig.update_layout(height=900, showlegend=True)
        return combined_fig
    
    fig.show()


def plot_multidimensional_3d(results, system_name, pp: int,  metrics_dict: dict , path: str = "" , save_html=False, show: bool = False  ):
    """
    Plot 3D trajectories of the system with different points per period and across different parameter values with interactive controls.
    
    Args:
        results: List of result dictionaries containing:
            - 'true_value': numpy array of true values (must be 3D)
            - 'predictions': torch tensor of predictions (must be 3D)
            - 'parameters': dictionary of parameters
        system_name: Name of the dynamical system
        save_html: Whether to save as interactive HTML file
        show: Whether to display the plot immediately
        metrics_dict: Standalone dictionary containing metrics and their values.
                      Can be structured as:
                      - {'RMSE': [0.1, 0.2], 'MAE': [0.01, 0.02]} OR
                      - {0: {'RMSE': 0.1}, 1: {'RMSE': 0.2}}
    
    Returns:
        plotly.graph_objects.Figure: Interactive 3D figure with dropdown
    """
    # Create figure
    fig = go.Figure()
    
    # Create visibility matrix and button definitions
    buttons = []
    visible_matrix = []
    param_strings = []   # To store formatted parameter strings
    metric_strings = []  # To store formatted metric strings
    
    for i, result in enumerate(results):
        true_vals = result['true_value']
        preds = result['predictions'].cpu().numpy() if hasattr(result['predictions'], 'cpu') else result['predictions']
        params = result['parameters']
        
        # Verify data is 3D
        if true_vals.shape[1] != 3 or preds.shape[1] != 3:
            raise ValueError("Data must be 3-dimensional (shape: [n_points, 3])")
        
        # 1. Parse out the metrics for the CURRENT run (index i) from your passed dictionary
        current_run_metrics = {}
        if metrics_dict:
            # Check if index-based structure: {0: {'RMSE': 0.1}, 1: {'RMSE': 0.2}}
            if i in metrics_dict and isinstance(metrics_dict[i], dict):
                current_run_metrics = metrics_dict[i]
            # Check if metric-based array structure: {'RMSE': [0.1, 0.2]}
            else:
                for metric_name, values in metrics_dict.items():
                    if isinstance(values, (list, tuple, np.ndarray)) and len(values) > i:
                        current_run_metrics[metric_name] = values[i]
                    elif isinstance(values, (int, float)): # Fallback for single-item runs
                        current_run_metrics[metric_name] = values

        # 2. Dynamically build the metric string (no hardcoded names!)
        metric_items = []
        for k, v in current_run_metrics.items():
            if isinstance(v, (int, float)):
                metric_items.append(f"{k}: {v:.3f}")
            else:
                metric_items.append(f"{k}: {v}")
        
        metric_str = ", ".join(metric_items)
        metric_strings.append(metric_str)
        metric_label_part = f" ({metric_str})" if metric_str else ""
        
        # Format parameters for display
        param_str = "<br>".join([f"{k}: {v}" for k, v in params.items()])
        param_strings.append(param_str)
        
        # Create visibility array for this parameter set
        visible = [False] * (len(results) * 2)  # (true + pred) * results
        
        # True values trace
        fig.add_trace(go.Scatter3d(
            x=true_vals[:, 0],
            y=true_vals[:, 1],
            z=true_vals[:, 2],
            name=f'True Trajectory {i+1}',
            marker=dict(size=2),
            line=dict(color='grey', width=6),
            visible=(i==0)  # Only show first set by default
        ))
        
        # Predictions trace
        fig.add_trace(go.Scatter3d(
            x=preds[:, 0],
            y=preds[:, 1],
            z=preds[:, 2],
            name=f'Predicted Trajectory {i+1}',
            marker=dict(size=2),
            line=dict(color='darkorange', width=7),
            visible=(i==0)
        ))
        
        # Set visibility for this parameter set
        visible[i*2] = True    # True values
        visible[i*2+1] = True  # Predictions
        
        visible_matrix.append(visible)
        
        # Create button for this parameter set
        buttons.append(dict(
            label=f"Params {i+1}{metric_label_part}",
            method="update",
            args=[{"visible": visible_matrix[i]},
                  {"title": {
                      "text": f"{system_name} - Set {i+1}{metric_label_part}<br><span style='font-size: 12px;'>{param_str}</span>",
                      "x": 0.5,
                      "xanchor": "center"
                  },
                  "scene": {  # Reset camera view when switching
                      "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 0.5}},
                      "xaxis": dict(showline=True, linecolor="black", linewidth=5, showgrid=False, zeroline=False, showticklabels=False, ticks="", title=""),
                      "yaxis": dict(showline=True, linecolor="black", linewidth=5, showgrid=False, zeroline=False, showticklabels=False, ticks="", title=""),
                      "zaxis": dict(showline=True, linecolor="black", linewidth=5, showgrid=False, zeroline=False, showticklabels=False, ticks="", title="")
                  }}]
        ))
    
    # Initial title with first parameter set
    initial_metric_part = f" ({metric_strings[0]})" if metric_strings and metric_strings[0] else ""
    initial_title = {
        "text": f"{system_name} - Set 1{initial_metric_part}<br><span style='font-size: 12px;'>{param_strings[0]}</span>",
        "x": 0.5,
        "xanchor": "center"
    }
    
    # Update layout with dropdown menu
    fig.update_layout(
        title=initial_title,
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.1,
            "y": 1.2,
            "xanchor": "left",
            "yanchor": "top"
        }],
        scene=dict(
            xaxis=dict(showline=True, linecolor="black", linewidth=5, showgrid=False, zeroline=False, showticklabels=False, ticks="", title=""),
            yaxis=dict(showline=True, linecolor="black", linewidth=5, showgrid=False, zeroline=False, showticklabels=False, ticks="", title=""),
            zaxis=dict(showline=True, linecolor="black", linewidth=5, showgrid=False, zeroline=False, showticklabels=False, ticks="", title=""),
            aspectmode='data',  # Preserve aspect ratio
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(t=120)
    )

    if save_html:
        fig.write_html(f"{path}{system_name}/{pp}.html")
        print(f"saved File at {path}{system_name}/{pp}.html")
    if show:
        fig.show()

    return fig


import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    
    df = pd.DataFrame(data)
    # Filter out exploded runs (NaN or Infinity) right at loading
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

# --- PLOT 1: 3D Parameter Space ---
def plot_3d_scatter(df, ax=None):
    """Plots a 3D scatter of SR, LR, and IS colored by RMSE."""
    show_plot = False
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        show_plot = True

    # Color by RMSE (Dark = Good/Low RMSE, Yellow = Bad/High RMSE)
    img = ax.scatter(df['SR'], df['LR'], df['IS'], c=df['RMSE'], 
                     cmap='viridis_r', alpha=0.8, edgecolors='k', linewidth=0.3)
    
    ax.set_xlabel('Spectral Radius')
    ax.set_ylabel('Leaky Rate')
    ax.set_title('3D Parameter Landscape (Color = RMSE)')
    
    # Handle colorbar carefully (attach to figure if possible, else ax)
    if ax.figure:
        ax.figure.colorbar(img, ax=ax, label='RMSE (Darker is Better)', shrink=0.6)
    
    if show_plot:
        plt.show()

def plot_parallel_coordinates(df, metric='RMSE', params=['SR', 'LR', 'IS'], ax=None, lower_is_better=True, save_path=None):
    """
    Plots parallel coordinates to show parameter flow, highlighting the best models.
    
    Args:
        df (pd.DataFrame): The dataframe containing results.
        metric (str): The column name to use for coloring and highlighting.
        params (list): List of parameter columns to plot on the x-axis.
        ax (plt.Axes, optional): Matplotlib axes to plot on.
        lower_is_better (bool): If True, treats lower values as 'best'.
        save_path (str, optional): File path to save the high-res image (e.g., 'parallel.png').
    """
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True

    # 1. Normalize data
    df_plot = df[params].copy()
    for col in params:
        if df_plot[col].max() == df_plot[col].min():
            df_plot[col] = 0.5 
        else:
            df_plot[col] = (df_plot[col] - df_plot[col].min()) / (df_plot[col].max() - df_plot[col].min())

    # 2. Attach metric
    df_plot['raw_metric'] = df[metric]
    
    # 3. Sort Data
    if lower_is_better:
        df_plot = df_plot.sort_values('raw_metric', ascending=False)
        threshold = df[metric].quantile(0.1)
        cmap = plt.cm.viridis_r  #type: ignore
    else:
        df_plot = df_plot.sort_values('raw_metric', ascending=True)
        threshold = df[metric].quantile(0.9)
        cmap = plt.cm.viridis #type: ignore

    m_min, m_max = df[metric].min(), df[metric].max()

    # 4. Plotting Loop
    for i, row in df_plot.iterrows():
        val = row['raw_metric']
        is_best = (val <= threshold) if lower_is_better else (val >= threshold)
        alpha = 0.9 if is_best else 0.05 
        
        # Color mapping
        norm_c = (val - m_min) / (m_max - m_min) if m_max > m_min else 0.5
        color = cmap(norm_c)
        
        ax.plot(params, row[params], color=color, alpha=alpha)

    # 5. Styling
    direction = "Lowest" if lower_is_better else "Highest"
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=m_min, vmax=m_max)) #type: ignore
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)

    if lower_is_better:
        cbar.ax.invert_yaxis()
        cbar.set_label(f'{metric} (Low = Best)')
    else:
        cbar.set_label(f'{metric} (High = Best)')
        
    ax.set_title(f'Parallel Coordinates: {metric} ({direction} 10% Highlighted)')
    ax.set_ylabel('Normalized Parameter Value (0-1)')
    ax.set_xlabel('Hyperparameters')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- SAVE LOGIC ---
    if save_path:
        # We access the figure from the axis to ensure we save what was drawn
        ax.figure.savefig(save_path, dpi=300, bbox_inches='tight') #type: ignore
        print(f"Parallel Coordinates plot saved to: {save_path}")

    if show_plot:
        plt.show()


# --- PLOT 2: Correlation Matrix ---
def plot_correlation_matrix(df, ax=None, save_path=None):
    """Plots heatmap of pairwise correlations."""
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        show_plot = True

    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix')
    
    # --- SAVE LOGIC ---
    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches='tight') #type: ignore
        print(f"Correlation Matrix saved to: {save_path}")

    if show_plot:
        plt.show()


# --- PLOT 3: Best vs Worst Distributions ---
def plot_parameter_distributions(df, metric='RMSE', params=['SR', 'LR', 'IS'], ax=None, lower_is_better=True, save_path=None):
    """
    Plots histograms of parameters for the top 20% performing models.
    """
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        show_plot = True

    # Filter for Top 20%
    if lower_is_better:
        threshold = df[metric].quantile(0.2)
        top_models = df[df[metric] <= threshold]
        direction_label = "Lowest"
    else:
        threshold = df[metric].quantile(0.8)
        top_models = df[df[metric] >= threshold]
        direction_label = "Highest"

    # Plot histograms
    for param in params:
        if param in df.columns:
            ax.hist(top_models[param], alpha=0.5, label=f'Best {param}', bins=10, density=True)
        else:
            print(f"Warning: Parameter '{param}' not found in DataFrame.")

    ax.legend()
    ax.set_title(f'Parameter Distribution for {direction_label} 20% {metric}')
    ax.set_ylabel('Density')
    ax.set_xlabel('Parameter Value')
    
    # --- SAVE LOGIC ---
    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches='tight') #type: ignore
        print(f"Distribution plot saved to: {save_path}")

    if show_plot:
        plt.show()

# --- MAIN DRIVER ---
def plot_all_analysis(df):
    """Creates a dashboard combining all 4 plots."""
    print(f"Plotting {len(df)} valid runs...")
    print("\n--- TOP 5 CONFIGURATIONS ---")
    print(df.sort_values('RMSE').head(5))

    fig = plt.figure(figsize=(18, 10))
    
    # Grid: 2 rows, 2 columns
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    plot_3d_scatter(df, ax=ax1)
    
    ax2 = fig.add_subplot(2, 2, 2)
    plot_parallel_coordinates(df, ax=ax2)
    
    ax3 = fig.add_subplot(2, 2, 3)
    plot_correlation_matrix(df, ax=ax3)
    
    ax4 = fig.add_subplot(2, 2, 4)
    plot_parameter_distributions(df, ax=ax4)
    
    plt.tight_layout()
    plt.show()









###______________________________ Abstractions ______________________________###


def _get_trace_builders(dim):
    """This function returns the number of traces, 1,2,3 for each dimension according to the dimension of the data.
    This is a abstraction is used because I do not want to reiterate the name setting everywhere in the function. Design Principle: Do NOT REPEAT YOURSELF

    Args:
        dim (_type_): Dimension of the data

    Raises:
        ValueError: If the dimension is not supported
    Returns:
        _type_: number of traces and 
    """
    if dim == 1 or dim == 2:
        def line_trace(x, y, name, color, lw): #type: ignore
            return go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(width=lw, color=color),
                name=name
            )

        def marker_trace(x, y, name, color, size): #type: ignore
            return go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(size=size, color=color),
                name=name,
                showlegend=False
            )

    elif dim == 3:
        def line_trace(x, y, z, name, color, lw):
            return go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(width=lw, color=color),
                name=name
            )

        def marker_trace(x, y, z, name, color, size):
            return go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(size=size, color=color),
                name=name,
                showlegend=False
            )

    else:
        raise ValueError("Unsupported dimension")

    return line_trace, marker_trace