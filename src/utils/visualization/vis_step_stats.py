from typing import Dict

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def byte_ytick_to_label(value, pos):
    """
    10, 100 stay as is, 1000 becomes 1K, 1000000 becomes 1M, etc.
    """
    if value < 1000:
        return str(int(value))
    elif value < 1_000_000:
        return f"{round(value / 1000)}K"
    elif value < 1_000_000_000:
        return f"{round(value / 1_000_000)}M"
    elif value < 1_000_000_000_000:
        return f"{round(value / 1_000_000_000)}G"
    else:
        return f"{round(value / 1_000_000_000_000)}T"


def delay_ytick_to_label(value, pos):
    """
    Converts delay values to a human-readable format.
    Values are provided in milliseconds (ms).
    """
    if value < 1000:
        return f"{round(value * 1000)} ms"
    else:
        return f"{round(value)} s"


def load_ytick_to_label(value, pos):
    """
    Converts load values to a human-readable format.
    Values are provided as percentages.
    """
    return f"{round(value * 100)}%"


def _vis_ep_step_stats(stat_info, plot_title, y_tick_format_func, annotate_ax=False, log_y_axis=False) -> Image:
    """
    Creates a matplotlib lineplot for the given statistics.

    Args:
        stat_info: A list of tuples, where each tuple contains:
            - stat_values: A list of values for the statistic.
            - stat_label: A label for the statistic to be used in the plot legend.
        plot_title: A string representing the title of the plot.
        num_steps: An integer representing the number of steps (x-axis limit).
        annotate_ax: A boolean indicating whether to annotate the last value of each statistic on the plot.

    Returns: A matplotlib lineplot including a line per provided stat, represented as an RGB image.
    """
    num_steps = len(stat_info[0][0])  # assuming all stat_values have the same length

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(plot_title, fontsize=20)

    #
    for i, (stat_values, stat_label) in enumerate(stat_info):
        # use colors from the colorblind palette
        color_i = plt.cm.tab10(i % 10)  # cycle through a set of colors
        ax.plot(list(range(1, num_steps + 1)), stat_values, label=stat_label, color=color_i, linewidth=2)
        if annotate_ax:
            stat_last_value = stat_values[-1]
            ax.annotate(byte_ytick_to_label(stat_last_value, None), xy=(num_steps, stat_last_value), color=color_i, fontsize=12)

    # set x axis limits and labels
    ax.set_xlim(1, num_steps)
    ax.set_xlabel('Episode step', fontsize=16)
    ax.set_xticks(list(range(1, num_steps + 1)))
    ax.set_xticklabels([str(i) for i in range(num_steps)], fontsize=14)

    # set y axis limits, labels and formatting
    if log_y_axis:
        ax.set_yscale('log')
    if y_tick_format_func == byte_ytick_to_label:
        ax.set_ylabel('Bytes', fontsize=16)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(byte_ytick_to_label))
    elif y_tick_format_func == delay_ytick_to_label:
        ax.set_ylabel('Delay', fontsize=16)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(delay_ytick_to_label))
    elif y_tick_format_func == load_ytick_to_label:
        ax.set_ylabel('Load', fontsize=16)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(load_ytick_to_label))
    else:
        raise ValueError("Unsupported y_tick_format_func provided.")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, loc='upper left')

    # draw the plot and convert it to an RGB image
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    buf = canvas.buffer_rgba()  # returns a memoryview of the RGBA buffer
    rgba = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
    ep_plot = Image.fromarray(rgba[:, :, :3], 'RGB')  # convert to RGB
    plt.close()

    return ep_plot


def visualize_ep_step_stats(ep_name, ep_stats) -> Dict[str, Image]:
    """
    Creates network performance plots concerning e.g. avg. packet delay, link utilization, packet stats.
    :return: A dictionary containing the RGB arrays of the created plots (keyed by their names)
    """
    ep_plot_dict = dict()

    # sent/inFlight/delivered/givenUp bytes: 'byte_stats'
    byte_stat_info = [
        (ep_stats["tx"], "sent app bytes"),
        (ep_stats["inFlight"], "in-flight app bytes"),
        (ep_stats["delivered"], "delivered app bytes"),
        (ep_stats["givenUp"], "dropped+lost app bytes"),
    ]
    ep_plot_dict[f'{ep_name}/step_stats_byte'] = _vis_ep_step_stats(byte_stat_info,
                                                                    f"{ep_name}: byte stats",
                                                                    byte_ytick_to_label,
                                                                    annotate_ax=True)
    # avg. delay: 'avg_delay'
    avg_delay_info = [
        (ep_stats['avgDelay'], "avg. packet delay"),
        (ep_stats['maxDelay'], "max. packet delay"),
    ]
    ep_plot_dict[f'{ep_name}/step_stats_delay'] = _vis_ep_step_stats(avg_delay_info,
                                                                     f"{ep_name}: delay stats",
                                                                     delay_ytick_to_label)
    # flow stats: 'flow_stats'
    flow_stat_info = [
        (ep_stats["udpRemainingFlowBytes"], "remaining bytes (UDP flows)"),
        (ep_stats["udpRemainingFlowDatarateSum"], "sum of datarates (UDP flows, per second)"),
        (ep_stats["tcpRemainingFlowBytes"], "remaining bytes (TCP flows)"),
        (ep_stats["tcpRemainingFlowCWndSum"], "sum of congestion windows (TCP flows)")
    ]
    ep_plot_dict[f'{ep_name}/step_stats_flow'] = _vis_ep_step_stats(flow_stat_info,
                                                                    f"{ep_name}: flow stats",
                                                                    byte_ytick_to_label,
                                                                    log_y_axis=True)
    # max LU and bandwidth usage per timestep: 'load'
    load_stat_info = [
        (ep_stats['maxLinkUtilization'], "max. LU"),
        (ep_stats['avgLinkUtilization'], "avg. LU"),
        (ep_stats['maxTxQueueRelativeLoad'], "max. tx queue load"),
        (ep_stats['avgTxQueueRelativeLoad'], "avg. tx queue load")
    ]
    ep_plot_dict[f'{ep_name}/step_stats_load'] = _vis_ep_step_stats(load_stat_info,
                                                                    f"{ep_name}: load stats",
                                                                    load_ytick_to_label)

    plt.close('all')
    return ep_plot_dict
