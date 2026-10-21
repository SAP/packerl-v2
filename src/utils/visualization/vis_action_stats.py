import io

from PIL import Image
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np


def visualize_action_stats(oscillation_ratio, next_hop_spread) -> Image:
    """
    Visualizes the action statistics of oscillation ratio and next hop spread.
    This function creates a combined plot with two subplots:
    1. Left: A heatmap (colored n by n matrix) of the oscillation ratio between nodes.
    2. Right: A heatmap (colored n by 1 matrix) of the next hop spread for each node.
    For both plots, we also write the respective values for the cells into the heatmap.

    Args:
        oscillation_ratio: A numpy array of shape (n, n) where n is the number of nodes, values ranging from 0 to 1.
                           The value of cell (i, j) represents the action oscillation ratio between nodes i and j.
        next_hop_spread: A numpy array of shape (n) where n is the number of nodes, values ranging from 0 to 1.
                         The value of cell (i) represents the next hop spread for node i.

    Returns: A PIL.Image object containing the combined plot.

    """
    n = oscillation_ratio.shape[0]

    # Create figure and gridspec layout: 1 row, 3 columns (matrix, vector, colorbar)
    # Set square cell size: each cell 0.7 inches tall, figure height proportional to n
    cell_size_inch = 0.7
    fig_height = n * cell_size_inch
    fig_width = fig_height * (1 + 1.5/n)  # width for matrix + vector + colorbar
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, 3, width_ratios=[n, 1, 0.2], wspace=0.1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    cbar_ax = fig.add_subplot(gs[2])

    # 1) Matrix heatmap with square cells
    _ = ax1.imshow(oscillation_ratio, cmap='viridis', vmin=0, vmax=1, aspect='equal')
    ax1.set_title('Action Oscillation Ratio')
    ax1.set_xlabel('Dest. Node Index')
    ax1.set_ylabel('Routing Node Index')
    ax1.set_xticks(np.arange(n))
    ax1.set_yticks(np.arange(n))
    for (i, j), val in np.ndenumerate(oscillation_ratio):
        ax1.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')

    # 2) Vector heatmap with square cells
    im2 = ax2.imshow(next_hop_spread.reshape(-1, 1), cmap='viridis', vmin=0, vmax=1, aspect='equal')
    ax2.set_title('Next Hop Spread')
    ax2.set_xticks([])
    ax2.set_yticks([])
    for i, val in enumerate(next_hop_spread):
        ax2.text(0, i, f'{val:.2f}', ha='center', va='center', color='white')

    # 3) Shared colorbar
    plt.colorbar(im2, cax=cbar_ax)

    # Export to PIL image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    action_stat_img = Image.open(buf).copy().convert('RGB')

    plt.close(fig)
    buf.close()

    return action_stat_img