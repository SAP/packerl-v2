from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt


seismic_cmap = plt.get_cmap('seismic')
seismic_cmap.set_bad(color='lightgray')

TITLE_PADDING = 15
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TEXT_FONTSIZE = 12
TEXT_NEWLINE_SPACING = 0.2
COLOR_MULTIPLIER = 3


def _fill_cell_text(ax, val_text, val_cell, val_min, val_max, val_mean, unit, val_is_relative, i, j,
                    subtext_mean=True):
    """
    Fill a cell in the matrix with the appropriately styled text.
    """
    with_subtext = j > 0 and not np.isnan(val_text)
    is_baseline = j == 0

    # determine text content and style
    val_prefix = "+" if not is_baseline and val_text > 0 and val_is_relative else ""  # add a plus sign for positive relative values
    if np.isnan(val_text):
        val_text = "N/A"
        text_color = 'black'
        text_weight = 'normal'
        unit_str = ""
    else:
        if val_is_relative and not is_baseline:
            val_text *= 100
            unit_str = '%'
        else:
            unit_str = unit
        val_text = int(val_text) if float(val_text).is_integer() else round(float(val_text), 2)  # shorten to 2 decimals
        text_color = 'black' if abs(val_cell) < 0.25 or is_baseline else 'white'  # white text for darker cells
        text_weight = 'bold' if is_baseline else 'normal'

    # place text and subtext
    text = f'{val_prefix}{val_text}{unit_str}'
    text_x = j
    text_y = i - TEXT_NEWLINE_SPACING if with_subtext else i
    ax.text(text_x, text_y, text, va='center', ha='center',
            color=text_color, weight=text_weight, fontsize=TEXT_FONTSIZE)
    if with_subtext:
        if val_min != val_max and not subtext_mean:
            val_min = int(val_min) if float(val_min).is_integer() else round(float(val_min), 2)
            val_max = int(val_max) if float(val_max).is_integer() else round(float(val_max), 2)
            subtext = f"({val_min}{unit}, {val_max}{unit})"
        else:
            val_mean = int(val_mean) if float(val_mean).is_integer() else round(float(val_mean), 2)
            subtext = f"({val_mean}{unit})"
        subtext_x = j
        subtext_y = i + TEXT_NEWLINE_SPACING
        ax.text(subtext_x, subtext_y, subtext, va='center', ha='center',
                color=text_color, fontsize=int(TEXT_FONTSIZE * 0.75))


def generate_cmap_cell_matrix(ax, results, metrics, units, metric_names, lower_is_better, baseline_name,
                              plot_title, plot_position,
                              extra_axvline=None,
                              always_show_xlabels=True,
                              always_show_ylabels=True,
                              subtext_mean=True):
    """
    Create a summary table with a heatmap-like color map.

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
    - plot_position: The position of the plot in the parent grid (influences axis visibility).

    Matrix plot specific parameters:
    - extra_axvline: An optional list of x-values for additional vertical lines.
    - always_show_xlabels: Whether to always show x-axis labels, or just for the first row.
    - always_show_ylabels: Whether to always show y-axis labels, or just for the first column.
    - subtext_mean: Whether to show the mean value in the cells' subtext, or the min/max values.
    """
    # First, check whether every metric has the same number of methods.
    num_methods = None
    for metric in metrics:
        if num_methods is None:
            num_methods = len(results[metric])
        elif len(results[metric]) != num_methods:
            raise ValueError("Every metric must have the same number of methods.")

    # Handle metrics: move the oscillation ratio to the first position if present
    if 'Mean oscillation_ratio_global' in metrics:
        metrics.remove('Mean oscillation_ratio_global')
        metrics = ['Mean oscillation_ratio_global'] + metrics

    # Handle methods: move the baseline method to the first position
    methods = list(results[metrics[0]].keys())
    assert baseline_name in methods, f"Baseline approach {baseline_name} not in the list of approaches {methods}"
    methods.remove(baseline_name)
    methods = [baseline_name] + methods

    # results_dict contains values per metric (1st level) and approach (2nd level) -> create a matrix.
    # if a metric or approach is not present for all approaches/metrics, raise an error
    results_iqm = deepcopy(results)
    results_min = deepcopy(results)
    results_max = deepcopy(results)
    for metric in metrics:
        for approach in methods:
            if approach not in results[metric]:
                raise ValueError(f"Approach {approach} not found for metric {metric}")
            if isinstance(results[metric][approach], list):
                # probabilistic approach: obtain the interquantile mean
                results_iqm[metric][approach] = np.mean(np.percentile(results[metric][approach], [25, 75]))
                results_min[metric][approach] = np.min(results[metric][approach])
                results_max[metric][approach] = np.max(results[metric][approach])
            else:
                # deterministic approach: copy the value to all result dicts
                results_iqm[metric][approach] = results[metric][approach]
                results_min[metric][approach] = results[metric][approach]
                results_max[metric][approach] = results[metric][approach]

    # sort the values and obtain result dicts for interquantile mean and min/max values
    data_means = np.array([[results_iqm[metric][approach] for approach in methods] for metric in metrics])
    data_mins = np.array([[results_min[metric][approach] for approach in methods] for metric in metrics])
    data_maxs = np.array([[results_max[metric][approach] for approach in methods] for metric in metrics])

    # obtain cell text data from means by subtracting the baseline value (except for the baseline column)
    data_text = data_means.copy()
    data_text[:, 1:] = data_text[:, 1:] - data_means[:, 0][:, np.newaxis]
    relative_rows = []
    for i, metric in enumerate(metrics):
        if units[metric] != "%" and data_text[i, 0] != 0:
            data_text[i, 1:] /= data_text[i, 0]
            relative_rows.append(True)
        else:
            relative_rows.append(False)

    # for cell styling data: flip the sign if lower is better
    data_cell = data_text.copy()
    data_cell[:, 0] = 0  # set baseline column to 0
    for j, metric in enumerate(metrics):
        if metric == 'oscillation_ratio_global':
            data_cell[j] = 0  # set oscillation ratio row to 0 (display as white)
        if not lower_is_better[metric]:
            data_cell[j] *= -1
        if units[metric] == "%":
            data_cell[j] /= 100  # since we show absolute changes for percentage values, divide color indicator by 100
    data_cell = np.where(data_cell > 0, data_cell * 1.2, data_cell)  # light blue is less salient -> amplify the values
    data_cell *= COLOR_MULTIPLIER  # amplify the color intensity

    # create the plot in given ax
    ax.set_title(plot_title, fontsize=TITLE_FONTSIZE, pad=TITLE_PADDING)
    plot_i, plot_j = plot_position
    if plot_i == 0 or always_show_xlabels:
        ax.get_xaxis().set_visible(True)
        plot_xticklabels = methods
    else:
        ax.get_xaxis().set_visible(False)
        plot_xticklabels = ["" for _ in methods]
    if plot_j == 0 or always_show_ylabels:
        ax.get_yaxis().set_visible(True)
        arrows = ['↓' if lower_is_better[metric] else '↑' for metric in metrics]
        plot_yticklabels = [f"{metric_names[metric]} ({arrow})" for metric, arrow in zip(metrics, arrows)]
    else:
        ax.get_yaxis().set_visible(False)
        plot_yticklabels = ["" for _ in metrics]
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.tick_params(labelbottom=False, labeltop=True, labelleft=True, labelright=False,
                   bottom=False, top=True, left=True, right=False)
    rows, cols = data_cell.shape
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(plot_xticklabels, fontsize=LABEL_FONTSIZE)
    ax.set_yticklabels(plot_yticklabels, fontsize=LABEL_FONTSIZE)

    # Visually separate the baseline column from the rest
    ax.axvline(x=0.5, color='black', linewidth=1)

    # Visually separate certain method groups if desired
    if extra_axvline is not None:
        if isinstance(extra_axvline, list):
            for line_pos in extra_axvline:
                ax.axvline(x=line_pos, color='black', linewidth=0.75)
        elif isinstance(extra_axvline, float):
            ax.axvline(x=extra_axvline, color='black', linewidth=0.75)
        elif isinstance(extra_axvline, int):
            ax.axvline(x=float(extra_axvline), color='black', linewidth=0.75)
        else:
            raise ValueError("extra_axvline must be a float, int or a list of floats")

    # Visually separate oscillation ratio metric
    if 'Mean oscillation_ratio_global' in metrics:
        ax.axhline(y=0.5, color='black', linewidth=0.5)

    # Add cell colors
    ax.matshow(data_cell, cmap=seismic_cmap, vmin=-1, vmax=1)
    ax.set_aspect(0.6)  # cell aspect ratio

    # Add cell text per metric and approach
    for i, metric in enumerate(metrics):
        unit = units if isinstance(units, str) else units[metric]
        val_is_relative = relative_rows[i]
        for j in range(cols):
            _fill_cell_text(ax,
                            data_text[i, j],
                            data_cell[i, j],
                            data_mins[i, j],
                            data_maxs[i, j],
                            data_means[i, j],
                            unit, val_is_relative, i, j,
                            subtext_mean=subtext_mean)
