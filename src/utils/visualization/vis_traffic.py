import io
from typing import List

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def visualize_traffic_matrices(traffic_matrices: List[np.ndarray]):
    """
    Visualize the given traffic matrices as a series of heatmaps.
    """
    max_traffic = np.max(np.stack(traffic_matrices, axis=0))
    visualized_frames = []

    for t, tm in enumerate(traffic_matrices):
        fig, ax = plt.subplots()
        cax = ax.matshow(tm, vmin=0, vmax=max_traffic)
        fig.colorbar(cax)
        ax.set_title(f"Traffic ({t=})")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        traffic_vis = Image.open(buf).copy().convert("RGB")   # copy so that the buffer can be closed
        visualized_frames.append(traffic_vis)

        plt.close(fig)
        buf.close()

    return visualized_frames
