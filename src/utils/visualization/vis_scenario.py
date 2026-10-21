import io

from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.scenario.events.traffic.flow_arrival import FlowArrival
from src.scenario.events.link_failures.link_failure import LinkFailure


def byte_ytick_to_label(value, pos):
    """
    10, 100 stay as is, 1000 becomes 1K, 1000000 becomes 1M, etc.
    """
    if value < 1000:
        return str(int(value))
    elif value < 1_000_000:
        return f"{int(value / 1000)}K"
    elif value < 1_000_000_000:
        return f"{int(value / 1_000_000)}M"
    elif value < 1_000_000_000_000:
        return f"{int(value / 1_000_000_000)}G"
    else:
        return f"{int(value / 1_000_000_000_000)}T"


def visualize_scenario_events(events) -> Image.Image:
    """
    Visualize the given list of scenario events.
    """
    flow_arrival_ts = []
    flow_amounts = []
    link_failure_data = []
    for events_t in events:
        for e in events_t:
            if isinstance(e, FlowArrival):
                flow_arrival_ts.append(e.t)
                flow_amounts.append(e.amount)
            elif isinstance(e, LinkFailure):
                link_failure_data.append((e.t, e.fst, e.snd))

    # Create the scenario plot
    plt.figure(figsize=(12, 9))
    plt.xlabel('Arrival time (ms)', fontsize=16)
    plt.ylabel('Amount (bytes)', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(byte_ytick_to_label))

    # plot flow arrival as scatter points (x-axis: arrival time, y-axis: amount)
    plt.scatter(flow_arrival_ts, flow_amounts, c='blue', label='Flow Arrivals')

    # plot link failures as vertical lines:
    #  - x-axis: time, y-axis: 0 to max amount
    #  - color: orange, linestyle: dashed
    #  - annotate with (fst, snd) at the top of the line
    for (t, fst, snd) in link_failure_data:
        plt.axvline(x=t, color='orange', linestyle='--')
        plt.annotate(f"({fst}, {snd})", fontsize=12, color='orange',
                     xy=(t, plt.ylim()[1]), xytext=(0, 2), textcoords='offset points',
                     ha='left', va='bottom')

    # add custom legend displaying a blue dot for flow arrivals and an orange dashed line for link failures
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='blue', markersize=10, label='Flow Arrivals'),
        plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label='Link Failures')
    ]
    plt.legend(handles=legend_handles, fontsize=14)

    # write the plot to a buffer and create an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    scenario_vis: Image.Image = Image.open(buf).copy().convert("RGB")  # copy so that the buffer can be closed

    # cleanup
    plt.close()
    buf.close()
    return scenario_vis
