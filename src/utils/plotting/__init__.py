from .radar_chart import generate_radar_chart, radar_factory
from .cmap_cell_matrix import generate_cmap_cell_matrix
from .single_metric_boxplot import generate_box_plots

import matplotlib.pyplot as plt
import seaborn as sns


def prepare_plots(plot_type, num_metrics):
    plot_materials = {}
    if plot_type == "matrix":
        pass
    elif plot_type == "boxplot":
        pass
    elif plot_type == "radar":
        plot_materials["theta"] = radar_factory(num_metrics)  # Create a radar chart type with polygonal grid lines.
    else:
        raise ValueError(f"Invalid plot type: {plot_type}")
    return plot_materials


def get_plot_func(plot_type):
    """
    Get the plotting function based on the plot type.
    """
    if plot_type == "matrix":
        return generate_cmap_cell_matrix
    elif plot_type == "boxplot":
        return generate_box_plots
    elif plot_type == "radar":
        return generate_radar_chart
    else:
        raise ValueError(f"Invalid plot type: {plot_type}")


def get_subplot_figsize(num_metrics, num_methods, plot_type):
    """
    Calculate the figure size based on the number of methods and metrics.
    """
    if plot_type == "matrix":
        figwidth = (num_methods + 2) * 1.2  # +2 for the metric names
        figheight = (num_metrics + 1.5) * 1.2  # +1 for the method names, +0.5 for the title
    elif plot_type == "boxplot":
        total_columns = 2 * num_metrics - 1  # gap between each boxplot
        width_ratios = [2 * num_methods if i % 2 == 0 else 1 for i in range(total_columns)]
        figwidth = sum(width_ratios)
        figheight = 5
    elif plot_type == "radar":
        figwidth = 8
        figheight = 8
    else:
        raise ValueError(f"Invalid plot type: {plot_type}")
    return figwidth, figheight


def get_subplot_kw(plot_type):
    """
    Get the subplot keyword arguments based on the plot type.
    """
    return {'projection': 'radar'} if plot_type == "radar" else None


def get_plot_kwargs(plot_type, plot_material, res_config, metrics, methods, num_cols, i):
    if plot_type == "matrix":
        return {
            "plot_position": (i // num_cols, i % num_cols),
            "extra_axvline": res_config.get("extra_axvline", None),
            "always_show_xlabels": res_config.get("always_show_xlabels", True),
            "always_show_ylabels": res_config.get("always_show_ylabels", True),
            "subtext_mean": res_config.get("subtext_mean", True),
        }
    elif plot_type == "boxplot":
        # Create a dictionary mapping from method to a unique color using seaborn's colorblind palette.
        reversed_colorblind_palette = sns.color_palette("colorblind", len(methods))[::-1]
        method_colors = dict(zip(methods, reversed_colorblind_palette))
        return {
            "method_colors": method_colors
        }
    elif plot_type == "radar":
        # Create a dictionary mapping from method to a unique color using seaborn's colorblind palette.
        reversed_colorblind_palette = sns.color_palette("colorblind", len(methods))[::-1]
        method_colors = dict(zip(methods, reversed_colorblind_palette))
        return {
            "method_colors": method_colors,
            "theta": plot_material['theta']
        }


# =====================================================================================================================


def generate_plots(results, res_config):
    """
    Create plots from the given results.

    Expects results to be a dict of result dicts:
    {
        "setup1": {
            "metric1": {
                "Method1": value,  # deterministic: one value
                "Method2": [value1, value2, ...],  # probabilistic: one value per random seed
                ...
            },
            "metric2": {
                ...
            },
            ...
        },
        "setup2": {
            ...
        },
        ...
    }
    For probabilistic methods, the list of values per metric is sorted wrt. overall performance (episode reward),
    and NOT wrt. the individual metrics! This means that the values for each metric are not necessarily sorted.

    Furthermore, expects a config dictionary 'res_config' that contains the following parameters:
    - setups: A list of setup names.
    - metrics: A list of metric names to be plotted.
    - units: A dictionary mapping metric names to their respective units.
    - lower_is_better: A dictionary mapping metric names to whether lower values are better.
    - methods: A list of method names.
    - baseline_name: The name of the deterministic baseline method.
    - num_rows: The number of rows in the plot grid.
    - num_cols: The number of columns in the plot grid.
    - plot_type: The type of plot to generate (e.g. one of "matrix", "boxplot", "radar").
    """
    metrics = res_config['metrics']
    num_metrics = len(metrics)
    metric_names = res_config['metric_names']
    units = res_config['units']
    lower_is_better = res_config['lower_is_better']
    methods = res_config['methods']
    num_methods = len(methods)
    baseline_name = res_config['baseline_name']
    num_rows = res_config['num_rows']
    num_cols = res_config['num_cols']

    # prepare plot grid
    plot_type = res_config['plot_type']
    plot_material = prepare_plots(plot_type, num_metrics)
    plot_func = get_plot_func(plot_type)  # get the correct plotting function based on the plot_type
    subplot_figsize = get_subplot_figsize(num_metrics, num_methods, plot_type)  # figsize depends on the plot type
    subplot_kw = get_subplot_kw(plot_type)  # get additional subplot kwargs depending on the plot type
    fig, axs = plt.subplots(num_rows, num_cols, figsize=subplot_figsize, subplot_kw=subplot_kw)
    axs_list = [axs] if isinstance(axs, plt.Axes) else axs.flatten()

    # fill the plots
    for i, setup in enumerate(res_config['setups']):
        setup_results = results[setup]
        plot_title = setup
        plot_kw = get_plot_kwargs(plot_type, plot_material, res_config, num_metrics, methods, num_cols, i) # get additional plot kwargs
        plot_func(axs_list[i], setup_results, metrics.copy(),
                  units, metric_names, lower_is_better, baseline_name, plot_title, **plot_kw)

    return fig
