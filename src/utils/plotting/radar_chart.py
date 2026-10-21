import matplotlib.pyplot as plt
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.path import Path
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
import numpy as np


def radar_factory(num_axes):
    """
    Create a radar chart with `num_axes` axes.
    The gridlines will be drawn as a polygon.
    """
    theta = np.linspace(0, 2 * np.pi, num_axes, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_axes)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.spines["polar"].set_visible(False)  # Hide default spines.

        def fill(self, *args, **kwargs):
            closed = kwargs.pop('closed', True)
            polygons = super().fill(closed=closed, *args, **kwargs)
            return polygons

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def _gen_axes_patch(self):
            verts = [(0.5 + 0.5 * np.cos(t), 0.5 + 0.5 * np.sin(t)) for t in theta]
            return plt.Polygon(verts, closed=True, edgecolor="k", facecolor="none")

        def draw(self, renderer):
            # Hide the default circular gridlines.
            for gl in self.yaxis.get_gridlines()[:-1]:
                gl.set_visible(False)
            self.yaxis.get_gridlines()[-1].set_linewidth(2)
            self.yaxis.get_gridlines()[-1].set_color("black")
            super().draw(renderer)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

    register_projection(RadarAxes)
    return theta


def generate_radar_chart(ax, results, metrics, units, metric_names, lower_is_better, baseline_name,
                         plot_title, theta, method_colors):
    """
    Create a radar chart for absolute values.

    The expected structure for `results` is:
    {
        "metric1": {
            "Method1": value,                  # deterministic: one value
            "Method2": [value1, value2, ...],    # probabilistic: one value per random seed
            ...
        },
        "metric2": { ... },
        ...
    }

    For probabilistic methods, each i-th entry in the list for every metric belongs to the same run.
    This function plots:
      - For deterministic methods: a thick line for the single value.
      - For probabilistic methods: each individual run as a thin, semi-transparent dashed line, and the mean
        over all runs as a thick line.

    Each spoke (metric) uses its own absolute value range. For metrics where lower is better,
    the axis is flipped so that smaller values are drawn toward the outer edge.
    """
    # Determine all algorithms across metrics.
    all_algs = set()
    for metric in metrics:
        for alg in results[metric].keys():
            all_algs.add(alg)
    if baseline_name in all_algs:
        all_algs.remove(baseline_name)
        all_algs = [baseline_name] + sorted(all_algs)
    else:
        raise ValueError("Baseline algorithm not found in results.")

    # Compute per-metric axis limits based on all algorithms and runs.
    axis_limits = {}
    for metric in metrics:
        values = []
        for alg in results[metric]:
            val = results[metric][alg]
            if isinstance(val, (list, np.ndarray)):
                values.extend(val)
            else:
                values.append(val)
        # In case min==max, adjust the range slightly.
        mn, mx = min(values), max(values)
        if mn == mx:
            mx = mn + 1
        axis_limits[metric] = (mn, mx)

    # Set custom labels for each spoke: here we drop the arrows since we plot absolute values.
    custom_labels = [f"{metric_names[metric]} [{units[metric]}]" for metric in metrics]
    ax.set_varlabels(custom_labels)

    # Remove default radial tick labels.
    ax.set_yticklabels([])

    # For each spoke, add tick marks and labels using the absolute value range.
    tick_length = 0.03  # length of tick marks in normalized coordinates
    for i, angle in enumerate(theta):
        metric = metrics[i]
        mn, mx = axis_limits[metric]
        ticks = np.linspace(mn, mx, 5)
        # Flip tick positions if lower is better.
        if lower_is_better.get(metric, False):
            norm_ticks = 1 - (ticks - mn) / (mx - mn)
        else:
            norm_ticks = (ticks - mn) / (mx - mn)
        for norm_val, tick_val in zip(norm_ticks, ticks):
            ax.text(angle, norm_val, f"{tick_val:.2f}", fontsize=6, ha='center', va='center')
            # Compute position in axes coordinates.
            R_axes = 0.5 * (norm_val + 1)
            x_center = 0.5 + R_axes * np.cos(angle)
            y_center = 0.5 + R_axes * np.sin(angle)
            dx = (tick_length / 2) * (-np.sin(angle))
            dy = (tick_length / 2) * (np.cos(angle))
            x1, y1 = x_center - dx, y_center - dy
            x2, y2 = x_center + dx, y_center + dy
            line = Line2D([x1, x2], [y1, y2], transform=ax.transAxes, lw=1)
            ax.add_line(line)

    # Set the overall radial limits to [0, 1] (after normalization per metric).
    ax.set_ylim(0, 1)

    # Plot each algorithm.
    for alg in all_algs:
        # Determine if this algorithm is probabilistic.
        first_val = results[metrics[0]][alg]
        is_probabilistic = isinstance(first_val, (list, np.ndarray))

        if is_probabilistic:
            num_runs = len(first_val)
            # Plot each individual run as a thin, semi-transparent dashed line.
            for run in range(num_runs):
                run_values = []
                for metric in metrics:
                    val = results[metric][alg][run]
                    mn, mx = axis_limits[metric]
                    if lower_is_better.get(metric, False):
                        norm_val = 1 - (val - mn) / (mx - mn)
                    else:
                        norm_val = (val - mn) / (mx - mn)
                    run_values.append(norm_val)
                # Close the polygon by appending the first value.
                run_values.append(run_values[0])
                ax.plot(np.append(theta, theta[0]), run_values, lw=0.5, linestyle='--',
                        color=method_colors[alg], alpha=0.5)
            # Compute the mean over runs for each metric.
            mean_values = []
            for metric in metrics:
                vals = np.array(results[metric][alg])
                mean_val = np.mean(vals)
                mn, mx = axis_limits[metric]
                if lower_is_better.get(metric, False):
                    norm_val = 1 - (mean_val - mn) / (mx - mn)
                else:
                    norm_val = (mean_val - mn) / (mx - mn)
                mean_values.append(norm_val)
            mean_values.append(mean_values[0])
            ax.plot(np.append(theta, theta[0]), mean_values, lw=3,
                    label=alg, color=method_colors[alg])
        else:
            # Deterministic: single value per metric.
            values = []
            for metric in metrics:
                val = results[metric][alg]
                mn, mx = axis_limits[metric]
                if lower_is_better.get(metric, False):
                    norm_val = 1 - (val - mn) / (mx - mn)
                else:
                    norm_val = (val - mn) / (mx - mn)
                values.append(norm_val)
            values.append(values[0])
            ax.plot(np.append(theta, theta[0]), values, lw=3,
                    label=alg, color=method_colors[alg])

    ax.set_title(plot_title, y=1.08, fontsize=16)
    ax.legend(loc='upper right')
