import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import numpy as np


yaxis_dev_colors = [(0.0, 0.0, 1.0),  # Blue
                    (0.5, 0.5, 0.5),  # Gray
                    (1.0, 0.0, 0.0)]  # Red

# Create a LinearSegmentedColormap from the list of colors.
yaxis_dev_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", yaxis_dev_colors, N=256)
yaxis_max_deviance = 10

grad_dev_colors = [(0.0, 0.0, 1.0),  # Blue
                   (1.0, 1.0, 1.0),  # White
                   (1.0, 0.0, 0.0)]  # Red
grad_dev_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", grad_dev_colors, N=256)
gradient_max_deviance = 20


def _generate_box_plot(ax, results, unit, metric_name, lower_is_better, method_colors, baseline_name):
    """
    Create a box plot for probabilistic methods with deterministic baseline references for a single metric.
    """
    # Separate deterministic and probabilistic approaches.
    det = {}
    prob = {}
    for alg, val in results.items():
        if isinstance(val, (list, np.ndarray)):
            prob[alg] = np.array(val)
        else:
            det[alg] = val

    # Obtain baseline value and calculate relative deviance for probabilistic methods.
    baseline = det[baseline_name]
    prob_names = list(prob.keys())
    data_list = []
    for alg in prob_names:
        data_list.append((np.array(prob[alg]) / baseline * 100 - 100))

    # Create box plot.
    positions = np.arange(1, len(prob_names) + 1)
    bp = ax.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True,
                    medianprops=dict(color="black"))
    # Set each box's color using the unique color from seaborn's colorblind palette.
    for patch, alg in zip(bp['boxes'], prob_names):
        patch.set_facecolor(method_colors[alg])
        patch.set_edgecolor("black")

    ax.set_xticks(positions)
    ax.set_xticklabels(prob_names, fontsize=12)
    for j, d in enumerate(data_list):
        x_vals = np.random.normal(loc=positions[j], scale=0.04, size=len(d))  # jitter in x-axis for visibility
        ax.scatter(x_vals, d, color=method_colors[prob_names[j]], alpha=0.6)

    # Plot deterministic approaches as horizontal lines.
    det_keys = sorted(det.keys(), key=lambda x: (x != baseline_name, x))
    for alg in det_keys:
        pct = (det[alg] / baseline * 100) - 100
        ax.axhline(pct, linestyle='--', color=method_colors[alg], label=f"{alg}")

    # left y-axis ticks: absolute values.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda t, pos: f"{baseline * (1 + t / 100):.2f}"))

    """
    # Highlight the baseline value in the y-axis.  TODO doesn't work as intended due to the formatter (uses the wrong limits)
    for ytick in ax.yaxis.get_ticklabels():
        if ytick.get_text() == f"{baseline:.2f}":
            ytick.set_color(method_colors[baseline_name])
            break
    """

    # Add left y-axis description as a text box (not as ylabel) above the axis.
    desc = f"Value [{unit}]"
    ax.text(0, 1.01, desc, fontsize=12, transform=ax.transAxes, rotation=0, ha='center', va='center')

    # Create secondary y-axis on the right for relative deviance.
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())

    # Set tick parameters so the labels are gray.
    ax2.tick_params(axis='y', labelcolor='gray')

    # Attach a dynamic formatter to display relative deviation values.
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda t, pos: f"{t:+.0f}%"))
    ax2.text(1, 1.01, f"vs. {baseline_name}", fontsize=12, color="gray",
             transform=ax.transAxes, rotation=0, ha='center', va='center')

    """
    # Set tick colors according to our deviance colormap. TODO doesn't work as intended due to the formatter (uses the wrong limits)
    for ytick in ax2.yaxis.get_ticklabels():
        val = float(ytick.get_text().replace("%", ""))
        # Cap at +/- max_deviance.
        val = max(-yaxis_max_deviance, min(yaxis_max_deviance, val))
        norm = (val + yaxis_max_deviance) / (2 * yaxis_max_deviance)
        norm = norm if lower_is_better else 1 - norm
        color = mcolors.to_hex(yaxis_dev_cmap(norm))
        ytick.set_color(color)
    """

    # --- Add a gradient background ---
    # It is important to get the updated x- and y-limits (after the box plot has been drawn).
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax2.get_ylim()
    n = 256
    rel_dev = np.clip(np.linspace(y_min, y_max, n), -gradient_max_deviance, gradient_max_deviance)
    # Normalize to [0, 1] using the same mapping as the tick labels.
    norm = (rel_dev + gradient_max_deviance) / (2 * gradient_max_deviance)
    if not lower_is_better:
        norm = 1 - norm
    gradient = norm.reshape(n, 1)  # make it a column vector
    # Display the gradient as an image over the entire plotting area.
    ax.imshow(gradient, extent=(x_min, x_max, y_min, y_max), origin='lower',
              aspect='auto', cmap=grad_dev_cmap, alpha=0.15, zorder=-1, vmin=0, vmax=1)
    # --------------------------------

    # Set title and legend.
    arrow = "(↓)" if lower_is_better else "(↑)"
    ax.set_title(f"\n{metric_name} {arrow}", fontsize=14, pad=5)
    ax.legend(fontsize=10, loc='upper right')


def generate_box_plots(ax, results, metrics, units, metric_names, lower_is_better, baseline_name, plot_title, method_colors):
    """
    Create box plots for probabilistic methods with deterministic baseline references.

    Expects results to be a dictionary with the following structure:
    {
        "metric1": {
            "Method1": value,  # deterministic: one value
            "Method2": [value1, value2, ...],  # probabilistic: one value per random seed
            ...
        },
        "metric2": {
            ...
        },
        ...
    }
    Furthermore, expects the following parameters:
    - metrics: A list of metric names to be plotted.
    - units: A dictionary mapping metric names to their respective units.
    - lower_is_better: A dictionary mapping metric names to whether lower values are better.
    - baseline_name: The name of the deterministic baseline method.
    - plot_title: The title of the entire plot.
    """
    num_metrics = len(metrics)
    num_methods = len(results[metrics[0]])
    total_columns = 2 * num_metrics - 1  # gap between each boxplot
    width_ratios = [2 * num_methods if i % 2 == 0 else 1 for i in range(total_columns)]

    # Capture the original parent and the axes position (so we can re-draw the axes and add a title)
    fig = ax.figure
    pos = ax.get_position()

    # Remove the original Axes so we can fill its area with a subgrid
    ax.remove()

    # Create a subgrid inside the area originally occupied by ax
    gs = gridspec.GridSpecFromSubplotSpec(1, total_columns,
                                          subplot_spec=ax.get_subplotspec(),
                                          width_ratios=width_ratios)
    # Create axes for the boxplots (using columns 0, 2, 4, etc.)
    sub_axes = [fig.add_subplot(gs[0, i * 2]) for i in range(num_metrics)]

    # Generate each individual boxplot into its corresponding sub-axis.
    for sub_ax, metric in zip(sub_axes, metrics):
        _generate_box_plot(sub_ax,
                           results[metric],
                           units[metric],
                           metric_names[metric],
                           lower_is_better[metric],
                           method_colors,
                           baseline_name)

    # Add the title centered above the subgrid area. We use the stored parent's position to determine the center.
    fig.text(pos.x0 + pos.width / 2, pos.y1 + 0.02, plot_title, ha='center', va='bottom', fontsize=16)
