import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

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

        if titles and i < len(titles):
            ax.set_title(titles[i])

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
            ax.set_title(titles[i])
        else:
            raise ValueError(f"Unsupported data dimension: {data.ndim}")

    plt.tight_layout()


def plot_components(trajectory, time=None, labels=None, title=None,
                           figsize=(10, 8), colors=None, linewidth=1.2,
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

    if title:
        fig.suptitle(title, y=1.02, fontsize=12)

    plt.xlabel('Time')
    plt.tight_layout()
