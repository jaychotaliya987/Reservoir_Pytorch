import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import matplotlib.cm as cm

def compare_plot(datasets, titles=None, figsize=(12, 8)):
    """
    Dynamically create subplots for datasets of varying dimensions.

    Args:
        datasets (list): List of numpy arrays (1D, 2D, or 3D)
        titles (list): Optional list of titles for each subplot
        figsize (tuple): Figure size
    """
    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()  # Convert to 1D array for easy indexing

    for i, data in enumerate(datasets):
        ax = axes[i]

        #if titles and i < len(titles):
        #    ax.set_title(titles[i])

        if data.ndim == 1:
            ax.plot(data)
        elif data.shape[1] == 2:
            if data.shape[1] == 2:  # If 2 columns, treat as x-y pairs
                ax.plot(data[:,0], data[:,1])
            else:  # Plot all columns
                for col in range(data.shape[1]):
                    ax.plot(data[:,col], label=f'Dim {col+1}')
                if data.shape[1] < 10:  # Only show legend if not too many lines
                    ax.legend()
        elif data.shape[1] == 3:
            # For 3D data, create projection

            ax.remove()
            ax = fig.add_subplot(1, n, i+1, projection='3d')
            ax.plot(data[:,0], data[:,1], data[:,2])
            #ax.set_title(titles[i])
        else:
            raise ValueError(f"Unsupported data dimension: {data.ndim}")

    plt.tight_layout()


def plot_components(trajectory, time=None, labels=None, title=None,
                           figsize=(10, 8), colors=None, linewidth=0.5,
                           spacing=0.3):
    """
    Stack trajectory components vertically with independent axes.

    Args:
        trajectory (np.ndarray): Shape (n_points, n_components)
        time (np.ndarray): Custom time values (default: indices)
        labels (list): Component names (e.g., ['x', 'y', 'z'])
        title (str): Overall title
        figsize (tuple): Figure size (width, height)
        colors (list): Line colors for each component
        linewidth (float): Line thickness
        spacing (float): Vertical space between subplots
        
    """
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    n_components = trajectory.shape[1]
    time = np.arange(trajectory.shape[0]) if time is None else time

    # Default settings
    if labels is None:
        labels = [f'{i}' for i in range(n_components)]
    if colors is None:
        colors = plt.cm.tab10.colors[:n_components]

    fig, axs = plt.subplots(n_components, 1, figsize=figsize,
                           sharex=False, sharey=False)

    if n_components == 1:
        axs = [axs]  # Ensure axs is iterable

    for i, ax in enumerate(axs):
        ax.plot(time, trajectory[:, i],
               color=colors[i],
               linewidth=linewidth)
        ax.set_ylabel(labels[i], rotation=0, ha='right', va='center')
        ax.grid(alpha=0.3)
        if i != 2:
            ax.set_xticklabels([])

        # Remove spines for cleaner look
        #ax.spines[['top', 'right']].set_visible(False)

    # Adjust spacing
    plt.subplots_adjust(hspace=spacing)

    if title!=None:
        plt.suptitle(f"Point Per Period: {title}")

    plt.xlabel('Time')
    plt.tight_layout()

def compare_plot_plotly(datasets, titles=None, figsize=(1920, 1080), colorscale='Viridis', 
                 line_width=3, marker_size=5, bgcolor='rgb(240, 240, 240)'):
    """
    Create beautiful interactive horizontal subplots for datasets of varying dimensions using Plotly.
    
    Args:
        datasets (list): List of numpy arrays (1D, 2D, or 3D)
        titles (list): Optional list of titles for each subplot
        figsize (tuple): Figure size (width, height)
        colorscale: Plotly colorscale name
        line_width: Width of plot lines
        marker_size: Size of start/end markers
        bgcolor: Background color
    """
    n = len(datasets)
    if titles is None:
        titles = [f'Dataset {i+1}' for i in range(n)]
    
    # Check if we need 3D subplots and create appropriate specs
    specs = []
    for data in datasets:
        if data.ndim > 1 and data.shape[1] == 3:
            specs.append({'type': 'scatter3d'})
        else:
            specs.append({'type': 'xy'})
    
    # Create horizontal subplots (1 row, n columns)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=titles,
        specs=[specs],  # Note: specs is now a list of dicts for columns
        horizontal_spacing=0.1 if n > 1 else 0.05
    )
    
    for i, data in enumerate(datasets):
        col = i+1
        
        if data.ndim == 1:
            # 1D data
            fig.add_trace(go.Scatter(
                y=data,
                mode='lines',
                line=dict(width=line_width, color=to_hex(cm.viridis(0.5))),
                name=f'Dataset {i+1}',
                hoverinfo='y'
            ), row=1, col=col)
            
        elif data.shape[1] == 2:
            # 2D data
            fig.add_trace(go.Scatter(
                x=data[:,0],
                y=data[:,1],
                mode='lines',
                line=dict(width=line_width, color='green'),
                name=f'Dataset {i+1}',
                hoverinfo='x+y'
            ), row=1, col=col)
            
            # Add start/end markers
            fig.add_trace(go.Scatter(
                x=[data[0,0]],
                y=[data[0,1]],
                mode='markers',
                marker=dict(size=marker_size, color='limegreen'),
                name='Start',
                showlegend=False,
                hoverinfo='none'
            ), row=1, col=col)
            
            fig.add_trace(go.Scatter(
                x=[data[-1,0]],
                y=[data[-1,1]],
                mode='markers',
                marker=dict(size=marker_size, color='crimson'),
                name='End',
                showlegend=False,
                hoverinfo='none'
            ), row=1, col=col)
            
        elif data.shape[1] == 3:
            # 3D data
            fig.add_trace(go.Scatter3d(
                x=data[:,0],
                y=data[:,1],
                z=data[:,2],
                mode='lines',
                line=dict(width=line_width, color=data[:,2], colorscale=colorscale),
                name=f'Dataset {i+1}',
                hoverinfo='x+y+z'
            ), row=1, col=col)
            
            # Add start/end markers
            fig.add_trace(go.Scatter3d(
                x=[data[0,0]],
                y=[data[0,1]],
                z=[data[0,2]],
                mode='markers',
                marker=dict(size=marker_size, color='limegreen'),
                name='Start',
                showlegend=False,
                hoverinfo='none'
            ), row=1, col=col)
            
            fig.add_trace(go.Scatter3d(
                x=[data[-1,0]],
                y=[data[-1,1]],
                z=[data[-1,2]],
                mode='markers',
                marker=dict(size=marker_size, color='crimson'),
                name='End',
                showlegend=False,
                hoverinfo='none'
            ), row=1, col=col)
            
            # Update 3D scene settings
            fig.update_scenes(
                aspectmode='data',
                xaxis=dict(gridcolor='rgb(200, 200, 200)', showbackground=True),
                yaxis=dict(gridcolor='rgb(200, 200, 200)', showbackground=True),
                zaxis=dict(gridcolor='rgb(200, 200, 200)', showbackground=True),
                bgcolor=bgcolor,
                row=1, col=col
            )
    
    # Update layout
    fig.update_layout(
        height=figsize[1],
        width=max(figsize[0], 300 * n),  # Ensure enough width for all subplots
        margin=dict(l=50, r=50, b=50, t=80),
        paper_bgcolor='white',
        plot_bgcolor=bgcolor,
        showlegend=False,
        font=dict(family='Arial', size=12))
    
    # Update subplot titles
    for i in range(n):
        fig.layout.annotations[i].update(x=0.5, xanchor='center', yanchor='bottom', 
                                      font=dict(size=14))
    
    fig.show()

def plot_components_plotly(trajectory, time=None, labels=None, title=None, 
                   figsize=(1920, 1080), colorscale='Viridis', line_width=2.5,
                   bgcolor='rgb(240, 240, 240)', title_fontsize=20):
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
    colors = [to_hex(cm.viridis(i/n_components)) for i in range(n_components)]
    
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
    fig.update_xaxes(title_text='Time', row=n_components, col=1)
    
    # Update subplot titles (component labels on left)
    for i in range(n_components):
        fig.layout.annotations[i].update(x=0.01, xanchor='left', font=dict(size=12))
    
    fig.show()



if __name__ == '__main__':
    t = np.linspace(0, 10, 500)
    data1 = np.column_stack([t, np.sin(t)])  # 2D
    data2 = np.column_stack([t, np.cos(t), np.sin(2*t)])  # 3D

# Create horizontal comparison plot
    compare_plot([data1, data2], 
                  titles=["Sine Wave", "3D Spiral"])
    plot_components(data2)
    
