import gc
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from tqdm import tqdm

# Configure professional logging pipeline
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def extract_metrics_from_folder(folder_path: Path) -> pd.DataFrame:
    """
    Parses all serialized pickle data files within a target directory 
    and extracts chaotic evaluation metrics into a structured DataFrame.

    Args:
        folder_path (Path): Path to the directory containing .pkl files.

    Returns:
        pd.DataFrame: Structured dataset containing parameter configurations,
                      discretization steps, and calculated error metrics.
    """
    # Deferred import to isolate external framework dependency
    try:
        from reservoirgrid.helpers import chaos_utils
    except ImportError as e:
        logger.error("Failed to import 'reservoirgrid'. Ensure it is accessible in the sys.path.")
        raise e

    all_data: List[Dict[str, Any]] = []
    file_list: List[Path] = sorted(list(folder_path.glob("*.pkl")))

    if not file_list:
        logger.warning(f"No configuration files found matching '*.pkl' in: {folder_path}")
        return pd.DataFrame()

    for file_path in tqdm(file_list, desc="Processing files", unit="file", leave=False):
        try:
            ppp = float(file_path.stem)
        except ValueError:
            logger.warning(f"Skipping file with non-numeric name: {file_path.name}")
            continue

        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, IOError) as e:
            logger.error(f"Failed to safely read file {file_path.name}: {str(e)}")
            continue

        for entry in data:
            try:
                params = entry["parameters"]
                true_val = entry["true_value"]
                preds = entry["predictions"]
                rmse = np.sqrt(np.mean((true_val - preds) ** 2))
                
                # Execute evaluation suite
                lyap_time = chaos_utils.lyapunov_time(truth=true_val, predictions=preds)
                kldiv = chaos_utils.kl_divergence(true_val, preds, bins=100)
                psd, cos_sim = chaos_utils.psd_metrics(true_val, preds)

                all_data.append({
                    "ppp": ppp,
                    "params": params,
                    "LyapunovTime": lyap_time,
                    "KLDivergence": kldiv,
                    "PSD Errors": psd,
                    "Cos_Sim": cos_sim,
                    "RMSE": rmse
                })
            except KeyError as ke:
                logger.warning(f"Malformed schema entry skipped in {file_path.name}: Missing key {str(ke)}")
                continue
        
        del data
        gc.collect()

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    # Generate an immutable, hashable identifier for unique hyperparameter states
    df["param_combo"] = df["params"].apply(lambda x: tuple(sorted(x.items())))
    return df


def plot_and_save_heatmap(df: pd.DataFrame, metric: str, folder_name: str, output_base_dir: Path) -> None:
    """
    Transforms long-form metrics data into a 2D matrix structure and saves a highly 
    polished, publication-ready heatmap visualization.

    Args:
        df (pd.DataFrame): Source evaluation metrics dataframe.
        metric (str): Target metric attribute column to plot.
        folder_name (str): Context identifier used for directory segmentation.
        output_base_dir (Path): Base directory path for visualization export.
    """
    # Reshape long-form layout into a highly optimized 2D grid matrix
    pivot_df = df.pivot(index="ppp", columns="param_combo", values=metric)
    pivot_df = pivot_df.sort_index(ascending=True)
    pivot_df = pivot_df.reindex(columns=sorted(pivot_df.columns))

    heatmap_matrix = pivot_df.to_numpy()
    
    # Extract structural components for ticks
    ppp_values = pivot_df.index.to_numpy()
    num_combos = heatmap_matrix.shape[1]

    # Select mathematical color norm strategy dynamically
    if metric in ["PSD Errors", "RMSE"]:
        norm = LogNorm(vmin=np.nanmin(heatmap_matrix[heatmap_matrix > 0]), vmax=np.nanmax(heatmap_matrix))
    elif metric == "Cos_Sim":
        norm = Normalize(vmin=-1.0, vmax=1.0)
    else:
        norm = Normalize(vmin=np.nanmin(heatmap_matrix), vmax=np.nanmax(heatmap_matrix))

    # Object-Oriented Canvas Configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(16, 7), layout="tight")
    
    cax = ax.imshow(
        heatmap_matrix,
        cmap="inferno",  # High perceptual uniformity for complex numerical spaces
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        norm=norm
    )

    # Clean, uncluttered X-Axis Ticks (Step value markers)
    x_tick_step = 5
    xticks = list(range(0, num_combos, x_tick_step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i) for i in xticks], fontsize=10, color="#2c3e50")
    ax.set_xlabel("Parameter Combination Index", fontsize=12, fontweight="bold", labelpad=12)

    # Exact Mapping Y-Axis Ticks (Actual PPP values instead of indices)
    y_tick_step = max(1, len(ppp_values) // 10)  # Dynamic step down-sampling for dense matrices
    yticks = list(range(0, len(ppp_values), y_tick_step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{ppp_values[i]:.2f}" for i in yticks], fontsize=10, color="#2c3e50")
    ax.set_ylabel("Points per Period (PPP)", fontsize=12, fontweight="bold", labelpad=12)

    # Colorbar Styling 
    cbar_formatter = ticker.LogFormatterMathtext() if isinstance(norm, LogNorm) else ticker.ScalarFormatter()
    cbar = fig.colorbar(cax, ax=ax, pad=0.02, fraction=0.046, format=cbar_formatter)
    cbar.set_label(label=f"Measured: {metric}", fontsize=11, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    # Master Frame Detailing
    ax.set_title(f"Evaluation Landscape: {metric} ({folder_name})", fontsize=14, fontweight="bold", pad=16, color="#1a252f")
    ax.spines[:].set_color("#bdc3c7")
    ax.grid(False)  # Remove overlay gridlines that obscure matrix pixels

    # Safe I/O Execution
    save_dir = output_base_dir / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f"{metric.lower().replace(' ', '_')}.png"
    
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_single_location(target_pkl_dir: str, output_base_dir: str = "Examples/Input_Discretization/Plots/HeatMaps/") -> None:
    """
    Orchestrates the extraction, normalization, and generation pipelines 
    for an isolated execution location directory.

    Args:
        target_pkl_dir (str): Relative or absolute target execution directory path.
        output_base_dir (str): Root export location path for output visual plots.
    """
    input_path = Path(target_pkl_dir)
    output_path = Path(output_base_dir)

    if not input_path.is_dir():
        logger.error(f"Execution terminated: Core target trajectory '{input_path}' is not an active directory structure.")
        return

    folder_name = input_path.name
    logger.info(f"Initiating processing loop for target location: {input_path}")

    # Extract long-form metrics frame
    df = extract_metrics_from_folder(input_path)
    if df.empty:
        logger.warning(f"Aborting execution: Zero target metric footprints extracted from {input_path.name}")
        return

    # Generate heatmaps sequentially
    target_metrics = ["LyapunovTime", "KLDivergence", "PSD Errors", "RMSE", "Cos_Sim"]
    for metric in target_metrics:
        logger.info(f"Generating heatmap projection matrix for metric: [{metric}]")
        plot_and_save_heatmap(df, metric, folder_name, output_path)

    del df
    gc.collect()
    logger.info(f"Processing sequence safely finalized for target domain: '{folder_name}'")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_single_location(sys.argv[1])
    else:
        logger.info("Direct execution trace missing target path parameter. Defaulting execution context example.")
        # Command-line alternative fallback placeholder:
        process_single_location("Examples/Input_Discretization/results/Chaotic/LorenzLHS")