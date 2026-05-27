import os
import pickle
from typing import Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
    
    # Use Plotly's built-in sample colorscale generator to completely avoid matplotlib imports
    colors = px.colors.sample_colorscale(colorscale, n_components)
    
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

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def compare_components(datasets, time=None, labels=None, component_labels=None, 
                       title=None, figsize=(1080, 700), line_width=2.5, 
                       bgcolor='rgb(240, 240, 240)', title_fontsize=20) -> go.Figure:
    """
    Create a subplot figure comparing the individual components of multiple datasets over time.
    Perfect for visualizing True vs. Predicted coordinate breakdowns (X, Y, Z) simultaneously.
    
    Args:
        datasets (list): List of numpy arrays. Each array has shape (n_points, n_components).
        time (np.ndarray): Custom time values or indices for the x-axis.
        labels (list): Labels for each dataset (e.g., ['True Trajectory', 'Predicted'])
        component_labels (list): Names for the components (e.g., ['X Component', 'Y Component'])
        title (str): Overall figure title
        figsize (tuple): Figure size (width, height)
        line_width: Width of the plot lines
        bgcolor: Background color for the plot area
        title_fontsize: Font size for the main title
        
    Returns:
        plotly.graph_objects.Figure
    """
    n_datasets = len(datasets)
    if n_datasets == 0:
        raise ValueError("The datasets list cannot be empty.")
        
    # Standardize 1D arrays to 2D column vectors
    processed_datasets = []
    for data in datasets:
        if data.ndim == 1:
            processed_datasets.append(data.reshape(-1, 1))
        else:
            processed_datasets.append(data)
            
    # Determine dimensions based on the first dataset
    n_points, n_components = processed_datasets[0].shape
    
    # Setup defaults if arguments aren't provided
    time = np.arange(n_points) if time is None else time
    
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(n_datasets)]
        
    if component_labels is None:
        component_labels = [f'Component {i+1}' for i in range(n_components)]
        
    # Qualitative color palette ensures distinct colors per dataset across all subplots
    colors = px.colors.qualitative.Plotly

    # Create stacked subplots (one row per component)
    fig = make_subplots(
        rows=n_components, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.06,  # Slightly loose spacing for clean labels
        subplot_titles=[f"<b>{label}</b>" for label in component_labels],
        specs=[[{'type': 'xy'}] for _ in range(n_components)]
    )

    # Loop through each component (row)
    for comp_idx in range(n_components):
        # Loop through each dataset (line trace within that row)
        for data_idx, data in enumerate(processed_datasets):
            
            # Check for potential shape mismatch across input datasets
            if data.shape[1] <= comp_idx:
                continue 
                
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=data[:, comp_idx],
                    mode='lines',
                    line=dict(width=line_width, color=colors[data_idx % len(colors)]),
                    name=labels[data_idx],
                    hoverinfo='x+y',
                    # Only show the legend item for the very first subplot row to avoid duplicates
                    showlegend=(comp_idx == 0) 
                ), 
                row=comp_idx + 1, 
                col=1
            )
            
        # Customize grid and axes for the subplot layer
        fig.update_yaxes(title_text=component_labels[comp_idx], row=comp_idx + 1, col=1)
        fig.update_xaxes(showgrid=True, gridcolor='rgb(215, 215, 215)', row=comp_idx + 1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='rgb(215, 215, 215)', row=comp_idx + 1, col=1)

    # Global layout adjustments
    fig.update_layout(
        height=figsize[1],
        width=figsize[0],
        title={
            'text': f"<b>{title}</b>" if title else "",
            'y': 0.96,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=title_fontsize, family='Arial')
        },
        margin=dict(l=60, r=50, b=60, t=100 if title else 60),
        paper_bgcolor='white',
        plot_bgcolor=bgcolor,
        hovermode='x unified', # Shows all dataset values simultaneously on hover
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(family='Arial', size=12)
    )

    # Label only the bottom-most x-axis to keep things perfectly neat
    fig.update_xaxes(title_text='Time', row=n_components, col=1)   
    
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
        hist_fig = px.histogram(
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
    
    fig.show

def plot_multidimensional_3d(results, system_name, pp: int, metrics_dict: dict, path: str = "", save_html=False, show: bool = False): 
    fig = go.Figure() 

    axis_config = dict( 
        showline=True, 
        linecolor="#1A202C", 
        linewidth=3, 
        showgrid=True, 
        gridcolor="#E2E8F0", 
        gridwidth=1, 
        zeroline=False, 
        showticklabels=False, 
        ticks="", 
        title="" 
    ) 

    js_metadata = []
    all_metric_keys = set()  # Collect unique metric names dynamically

    # --- PASS 1: Add all traces & compile clean metadata --- 
    for i, result in enumerate(results): 
        true_vals = result['true_value'] 
        preds = result['predictions'].cpu().numpy() if hasattr(result['predictions'], 'cpu') else result['predictions'] 
        params = result['parameters'] 

        if true_vals.shape[1] != 3 or preds.shape[1] != 3: 
            raise ValueError("Data must be 3-dimensional (shape: [n_points, 3])") 

        current_run_metrics = {} 
        if metrics_dict: 
            if i in metrics_dict and isinstance(metrics_dict[i], dict): 
                current_run_metrics = metrics_dict[i] 
            else: 
                for metric_name, values in metrics_dict.items(): 
                    if isinstance(values, (list, tuple, np.ndarray)) and len(values) > i: 
                        current_run_metrics[metric_name] = values[i] 
                    elif isinstance(values, (int, float)): 
                        current_run_metrics[metric_name] = values 

        # Track keys for our dropdown list generator
        for k in current_run_metrics.keys():
            all_metric_keys.add(k)

        metric_items = [ 
            f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
            for k, v in current_run_metrics.items() 
        ] 
        metric_str = " | ".join(metric_items)
        param_str = "  ·  ".join([f"<b>{k}</b>: {v}" for k, v in params.items()]) 

        # Map metrics directly to this structured object for the browser
        js_metadata.append({
            "trace_pair_index": i,
            "label": f"Set {i+1}",
            "params": params,       
            "metrics": current_run_metrics,  # CHANGE: Now sending metrics for sorting!
            "param_str": f"<span style='color:#718096; font-size:11px;'>{param_str}</span>",
            "metric_str": metric_str
        })

        fig.add_trace(go.Scatter3d( 
            x=true_vals[:, 0], y=true_vals[:, 1], z=true_vals[:, 2], 
            name='True Trajectory', 
            mode='lines', 
            line=dict(color='#4A5568', width=4), 
            visible=(i == 0) 
        )) 
        fig.add_trace(go.Scatter3d( 
            x=preds[:, 0], y=preds[:, 1], z=preds[:, 2], 
            name='Predicted Trajectory', 
            mode='lines', 
            line=dict(color='#FF6B6B', width=5.5), 
            visible=(i == 0) 
        )) 

    fig.update_layout( 
        title=dict( 
            text=f"<b>{system_name}</b>", 
            x=0.02, xanchor="left", y=0.98, yanchor="top", 
            font=dict(size=16, color="#2D3748") 
        ), 
        annotations=[dict( 
            text=js_metadata[0]['param_str'], 
            x=0.02, y=1.1, 
            xref="paper", yref="paper", xanchor="left", yanchor="top", 
            showarrow=False, align="left", 
        )], 
        font=dict(family="Inter, BlinkMacSystemFont, Segoe UI, sans-serif", size=12, color="#2D3748"), 
        scene=dict( 
            xaxis=axis_config, yaxis=axis_config, zaxis=axis_config, 
            aspectmode='data', 
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=0.6), up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=-0.1)
            ) 
        ), 
        legend=dict( 
            orientation="h", yanchor="top", y=1, xanchor="right", x=0.99, 
            font=dict(size=11), bgcolor="rgba(255,255,255,0.8)", bordercolor="#CBD5E0", borderwidth=1, 
        ), 
        template="plotly_white", 
        margin=dict(t=140, b=40, l=30, r=30), 
        height=750 
    ) 

    # --- JAVASCRIPT INJECTION ---
    js_data_serialized = json.dumps(js_metadata)
    metric_keys_serialized = json.dumps(list(all_metric_keys))

    custom_js = f"""
    const gd = document.getElementById('{{plot_id}}');
    
    const controlDiv = document.createElement('div');
    controlDiv.style.position = 'absolute';
    controlDiv.style.top = '45px';
    controlDiv.style.left = '20px';
    controlDiv.style.zIndex = '1000';
    controlDiv.style.display = 'flex';
    controlDiv.style.gap = '10px';
    controlDiv.style.fontFamily = 'Inter, sans-serif';
    controlDiv.style.fontSize = '12px';

    // Build metric sorting options dynamically
    let sortOptionsHtml = '<option value="default">Default Order (Index)</option>';
    const metricKeys = {metric_keys_serialized};
    metricKeys.forEach(k => {{
        sortOptionsHtml += `<option value="${{k}}">Sort by: ${{k}} (Asc)</option>`;
    }});

    controlDiv.innerHTML = `
        <div>
            <label style="display:block; color:#718096; margin-bottom:2px;">Sort Metrics:</label>
            <select id="js-sort-by" style="padding:4px 8px; border:1px solid #CBD5E0; border-radius:4px; background:#fff; min-width:180px;">
                ${{sortOptionsHtml}}
            </select>
        </div>
        <div>
            <label style="display:block; color:#718096; margin-bottom:2px;">Select Run Trace:</label>
            <select id="js-trace-selector" style="padding:4px 8px; border:1px solid #CBD5E0; border-radius:4px; background:#fff; min-width:280px; font-size:11px;">
            </select>
        </div>
    `;
    gd.appendChild(controlDiv);

    let datasets = {js_data_serialized};
    const sortSelect = document.getElementById('js-sort-by');
    const traceSelect = document.getElementById('js-trace-selector');

    function rebuildDropdown() {{
        const sortBy = sortSelect.value;
        
        if (sortBy !== 'default') {{
            const isDesc = sortBy.endsWith('_desc');
            const key = isDesc ? sortBy.slice(0, -5) : sortBy;
            
            datasets.sort((a, b) => {{
                let valA = a.metrics[key];
                let valB = b.metrics[key];
                
                // Fallbacks if a run is missing a specific metric key
                if (valA === undefined) return 1;
                if (valB === undefined) return -1;
                
                return isDesc ? (valB - valA) : (valA - valB);
            }});
        }} else {{
            datasets.sort((a, b) => a.trace_pair_index - b.trace_pair_index);
        }}

        traceSelect.innerHTML = '';
        datasets.forEach(d => {{
            const mPart = d.metric_str ? ` (${{d.metric_str}})` : '';
            const option = document.createElement('option');
            option.value = d.trace_pair_index;
            option.innerText = `${{d.label}}${{mPart}}`;
            traceSelect.appendChild(option);
        }});
        
        updatePlot(datasets[0].trace_pair_index);
    }}

    function updatePlot(activeTargetIdx) {{
        const totalTraces = gd.data.length;
        const visibleArray = new Array(totalTraces).fill(false);
        
        visibleArray[activeTargetIdx * 2] = true;
        visibleArray[activeTargetIdx * 2 + 1] = true;

        const targetData = datasets.find(d => d.trace_pair_index === activeTargetIdx);

        Plotly.update(gd, 
            {{ 'visible': visibleArray }}, 
            {{ 'annotations[0].text': targetData.param_str }}
        );
    }}

    sortSelect.addEventListener('change', rebuildDropdown);
    traceSelect.addEventListener('change', (e) => updatePlot(parseInt(e.target.value)));

    rebuildDropdown();
    """

    if save_html: 
        full_path = os.path.join(path, system_name) 
        os.makedirs(full_path, exist_ok=True) 
        file_loc = os.path.join(full_path, f"{pp}.html") 
        fig.write_html(file_loc, post_script=custom_js) 
        print(f"Saved custom metric-sort layout successfully at {file_loc}") 

    if show: 
        fig.show() 

    return fig

def load_and_parse(path):
    print(f"Loading {path}...")
    with open(path, 'rb') as f:
        results = pickle.load(f)
    
    data = []
    for res in results:
        row = {
            'SR': res['parameters']['SpectralRadius'],
            'LR': res['parameters']['LeakyRate'],
            'IS': res['parameters']['InputScaling'],
            'RMSE': res['metrics']['RMSE']
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def plot_3d_scatter(df, ax=None):
    show_plot = False
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        show_plot = True

    img = ax.scatter(df['SR'], df['LR'], df['IS'], c=df['RMSE'], 
                     cmap='viridis_r', alpha=0.8, edgecolors='k', linewidth=0.3)
    
    ax.set_xlabel('Spectral Radius')
    ax.set_ylabel('Leaky Rate')
    ax.set_title('3D Parameter Landscape (Color = RMSE)')
    
    if ax.figure:
        ax.figure.colorbar(img, ax=ax, label='RMSE (Darker is Better)', shrink=0.6)
    
    if show_plot:
        plt.show()

def plot_parallel_coordinates(df, metric='RMSE', params=['SR', 'LR', 'IS'], ax=None, lower_is_better=True, save_path=None):
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True

    df_plot = df[params].copy()
    for col in params:
        if df_plot[col].max() == df_plot[col].min():
            df_plot[col] = 0.5 
        else:
            df_plot[col] = (df_plot[col] - df_plot[col].min()) / (df_plot[col].max() - df_plot[col].min())

    df_plot['raw_metric'] = df[metric]
    
    if lower_is_better:
        df_plot = df_plot.sort_values('raw_metric', ascending=False)
        threshold = df[metric].quantile(0.1)
        cmap = plt.cm.viridis_r  #type: ignore
    else:
        df_plot = df_plot.sort_values('raw_metric', ascending=True)
        threshold = df[metric].quantile(0.9)
        cmap = plt.cm.viridis #type: ignore

    m_min, m_max = df[metric].min(), df[metric].max()

    for i, row in df_plot.iterrows():
        val = row['raw_metric']
        is_best = (val <= threshold) if lower_is_better else (val >= threshold)
        alpha = 0.9 if is_best else 0.05 
        
        norm_c = (val - m_min) / (m_max - m_min) if m_max > m_min else 0.5
        color = cmap(norm_c)
        
        ax.plot(params, row[params], color=color, alpha=alpha)

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

    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches='tight') #type: ignore
        print(f"Parallel Coordinates plot saved to: {save_path}")

    if show_plot:
        plt.show()


def plot_correlation_matrix(df, ax=None, save_path=None):
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        show_plot = True

    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation Matrix')
    
    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches='tight') #type: ignore
        print(f"Correlation Matrix saved to: {save_path}")

    if show_plot:
        plt.show()


def plot_parameter_distributions(df, metric='RMSE', params=['SR', 'LR', 'IS'], ax=None, lower_is_better=True, save_path=None):
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        show_plot = True

    if lower_is_better:
        threshold = df[metric].quantile(0.2)
        top_models = df[df[metric] <= threshold]
        direction_label = "Lowest"
    else:
        threshold = df[metric].quantile(0.8)
        top_models = df[df[metric] >= threshold]
        direction_label = "Highest"

    for param in params:
        if param in df.columns:
            ax.hist(top_models[param], alpha=0.5, label=f'Best {param}', bins=10, density=True)
        else:
            print(f"Warning: Parameter '{param}' not found in DataFrame.")

    ax.legend()
    ax.set_title(f'Parameter Distribution for {direction_label} 20% {metric}')
    ax.set_ylabel('Density')
    ax.set_xlabel('Parameter Value')
    
    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches='tight') #type: ignore
        print(f"Distribution plot saved to: {save_path}")

    if show_plot:
        plt.show()


def plot_all_analysis(df):
    """Creates a dashboard combining all 4 plots."""
    print(f"Plotting {len(df)} valid runs...")
    print("\n--- TOP 5 CONFIGURATIONS ---")
    print(df.sort_values('RMSE').head(5))

    fig = plt.figure(figsize=(18, 10))
    
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