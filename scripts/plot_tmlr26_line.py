import argparse
from collections import defaultdict
from datetime import datetime
import importlib
import math
from numbers import Real
from pathlib import Path
from statistics import mean, median
import sys
sys.path.append('.')
from typing import List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

from src.utils.constants import DEFAULT_OUT_DIR
from scripts._plot_utils import crawl_results, BASELINE_COLORS, BASELINE_DASHES, BASELINE_LEGEND_NAMES


def get_iqm(values: List[float]) -> float:
    """Compute the interquartile mean (IQM) of a list of values."""
    if not values:
        raise ValueError("Cannot compute IQM of an empty list.")
    sorted_values = sorted(values)
    n = len(sorted_values)
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    if n % 4 == 0:
        iqm_values = sorted_values[q1_idx:q3_idx]
    else:
        iqm_values = sorted_values[q1_idx:q3_idx + 1]
    return mean(iqm_values)


reducer_func = {
    "none": lambda x: x,   # no reduction, keep all replicates for boxplot
    "mean": mean,          # reduce replicates to their mean for line plot
    "median": median,      # reduce replicates to their median for line plot
    "iqm": get_iqm,        # reduce replicates to their interquartile mean for line plot
    # can add more reducers here if needed (e.g. median)
}

def get_run_matrix(all_runs, args: argparse.Namespace, exp_module, reducer="iqm"):

    all_lines = exp_module.all_lines
    all_plots = exp_module.all_plots
    all_hyper = exp_module.x_ticks
    hyper_values = exp_module.x_tick_values

    line_names = [name for name, _ in all_lines]
    plot_names = [name for name, _ in all_plots]
    hyper_name = exp_module.x_tick_name
    metric_name = exp_module.metric_name

    reducer_fn = reducer_func.get(reducer)
    if reducer_fn is None:
        raise ValueError(
            f"Reducer {reducer!r} not recognized. "
            f"Valid options are: {list(reducer_func.keys())}"
        )

    # (plot_idx, line_idx) -> {hyper_value: [metric, ...]}
    grouped = defaultdict(lambda: defaultdict(list))

    for r in all_runs:
        # filter out runs that did not finish
        if metric_name not in r.summary:
            if args.debug:
                print(f"Run {r.id} did not finish.")
            continue

        cfg = dict(r.config)
        summary = r.summary
        metric = summary.get(metric_name)
        if metric is None:
            if args.debug:
                print(f"Run {r.id} does not have metric {metric_name} in summary.")
            continue

        line_idxs = [i for i, (_, f) in enumerate(all_lines) if f(cfg)]
        plot_idxs = [i for i, (_, f) in enumerate(all_plots) if f(cfg)]
        hyper_idxs = [i for i, f in enumerate(all_hyper) if f(cfg)]

        if not line_idxs or not plot_idxs or not hyper_idxs:
            if args.debug:
                print(
                    f"Run {r.id} did not match line/plot/hyper filters: "
                    f"line_idxs={line_idxs}, plot_idxs={plot_idxs}, hyper_idxs={hyper_idxs}"
                )
            continue

        for line_idx in line_idxs:
            for plot_idx in plot_idxs:
                for hyper_idx in hyper_idxs:
                    h = hyper_values[hyper_idx]
                    grouped[(plot_idx, line_idx)][h].append(metric)

    # reduce replicates
    plot_data = {}
    for key, hyper_dict in grouped.items():
        xs = sorted(hyper_dict)
        ys = [reducer_fn(hyper_dict[x]) for x in xs]
        plot_data[key] = (xs, ys)

    return plot_data, plot_names, line_names, hyper_name, metric_name


def create_1metric_plots(run_matrix, plot_names, line_names, hyper_name, metric_name, exp_module):
    n_plots = len(plot_names)
    hyper_values = list(exp_module.x_tick_values)

    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(3 * n_plots, 2.5),
        squeeze=False,
        sharey=True,
    )
    axes = axes[0]

    x_min = min(hyper_values)
    x_max = max(hyper_values)

    baseline_name, baseline_val = exp_module.baseline
    baseline_color = BASELINE_COLORS.get(baseline_name, "gray")
    baseline_dash = BASELINE_DASHES[baseline_name]

    for plot_idx, ax in enumerate(axes):
        plot_has_data = False

        line_info = zip(line_names, exp_module.line_colors, exp_module.line_markers, exp_module.line_dashes)
        for line_idx, (line_name, line_color, line_marker, line_dash) in enumerate(line_info):
            key = (plot_idx, line_idx)
            if key not in run_matrix:
                continue

            xs, ys = run_matrix[key]
            if not xs:
                continue

            if ys and isinstance(ys[0], (list, tuple)):
                raise ValueError(
                    "create_1metric_plots received unreduced replicate lists in ys. "
                    "Use reducer='mean' before plotting line plots."
                )

            ax.plot(xs, ys, color=line_color, marker=line_marker, linewidth=2, markersize=5,
                    label=line_name, dashes=line_dash or (1,0))
            plot_has_data = True

        ax.set_title(f"{plot_names[plot_idx]} evaluation")
        #ax.set_xlabel(hyper_name)
        ax.grid(True, alpha=0.3)

        if plot_idx == 0:
            ax.set_ylabel(metric_name.split("/")[-1])

        # force full hyperparameter range
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(hyper_values)

        # Y-axis limits
        if hasattr(exp_module, "scales_boxplot"):
            vmin, vmax = exp_module.scales_boxplot
            ax.set_ylim(bottom=vmin, top=vmax)

        # baseline
        ax.axhline(
            y=baseline_val,
            color=baseline_color,
            linestyle="--",
            linewidth=2,
            dashes=baseline_dash,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    baseline_handle = Line2D(
            [0],
            [0],
            color=BASELINE_COLORS.get(baseline_name, "gray"),
            linestyle="--",
            linewidth=2,
            label=baseline_name,
            dashes=BASELINE_DASHES[baseline_name],
        )
    handles += [baseline_handle]
    labels += [baseline_name]

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            #bbox_to_anchor=(0.5, 0.98),
            ncol=max(1, len(handles)),
            frameon=True,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def create_tikz_code(run_matrix, plot_names, line_names, hyper_name, metric_name, exp_module):
    """
    Return TikZ/pgfplots code for the same multi-panel line plot layout as
    create_1metric_plots().

    Requires in LaTeX:
        \\usepackage{pgfplots}
        \\usepgfplotslibrary{groupplots}
        \\pgfplotsset{compat=1.18}

    Notes:
    - Uses one horizontal groupplot with shared y-limits.
    - Reuses axis limits, xticks, titles, ylabel, colors, dashes, and markers
      as closely as practical in pgfplots.
    - Legend is placed centered above the full group, similar to fig.legend(..., loc="upper center").
    """
    def _fmt_num(x):
        if not isinstance(x, Real) or isinstance(x, bool):
            raise TypeError(f"Expected a real number, got {type(x).__name__}: {x!r}")
        x = float(x)
        if not math.isfinite(x):
            raise ValueError(f"Non-finite numeric value encountered: {x!r}")
        return f"{x:.12g}"

    def _normalize_dash_pattern(dash):
        """
        Convert matplotlib-style dash specs to a pgfplots/TikZ dash option.

        Supports:
        - None / "" / "solid"
        - strings like "--", "-.", ":", "dashed", "dashdot", "dotted"
        - tuple/list of on/off lengths, e.g. (4, 2)
        - matplotlib full format (offset, (on, off, ...))
        """
        if dash is None or dash == "" or dash == "solid":
            return None

        if isinstance(dash, str):
            mapping = {
                "-": None,
                "--": "dashed",
                "-.": "dash dot",
                ":": "dotted",
                "dashed": "dashed",
                "dashdot": "dash dot",
                "dotted": "dotted",
            }
            return mapping.get(dash, None)

        if isinstance(dash, (tuple, list)):
            seq = list(dash)

            # matplotlib can store as (offset, (on, off, ...))
            if len(seq) == 2 and isinstance(seq[1], (tuple, list)):
                seq = list(seq[1])

            if not seq:
                return None

            seq = [float(v) for v in seq if float(v) > 0]
            if not seq:
                return None

            # Treat patterns like (1, 0) as solid
            if len(seq) == 2 and seq[1] == 0:
                return None

            if len(seq) % 2 == 1:
                seq = seq * 2

            parts = []
            for i, v in enumerate(seq):
                parts.append(("on " if i % 2 == 0 else "off ") + f"{_fmt_num(v)}pt")
            return "dash pattern=" + " ".join(parts)

        return None

    def _marker_to_pgf(marker):
        """
        Approximate matplotlib markers in pgfplots.
        """
        mapping = {
            None: "none",
            "": "none",
            "None": "none",
            " ": "none",
            ",": "none",
            ".": "*",
            "o": "*",
            "s": "square*",
            "^": "triangle*",
            "v": "triangle*",
            "<": "triangle*",
            ">": "triangle*",
            "D": "diamond*",
            "d": "diamond*",
            "p": "pentagon*",
            "*": "asterisk",
            "h": "star",
            "H": "star",
            "+": "+",
            "x": "x",
            "1": "triangle*",
            "2": "triangle*",
            "3": "triangle*",
            "4": "triangle*",
            "|": "|",
            "_": "_",
        }
        return mapping.get(marker, "*")

    def _collect_color_defs():
        """
        Build \\definecolor entries for all line colors plus the baseline color.
        """
        color_defs = []
        color_name_map = {}
        next_idx = 0

        def register(color_spec):
            nonlocal next_idx
            key = repr(color_spec)
            if key in color_name_map:
                return color_name_map[key]
            r, g, b = to_rgb(color_spec)
            R = int(round(255 * r))
            G = int(round(255 * g))
            B = int(round(255 * b))
            name = f"plotcolor{next_idx}"
            next_idx += 1
            color_name_map[key] = name
            color_defs.append(rf"\definecolor{{{name}}}{{RGB}}{{{R},{G},{B}}}")
            return name

        # line colors
        for c in exp_module.line_colors:
            register(c)

        # baseline color
        baseline_name, _baseline_val = exp_module.baseline
        baseline_color = BASELINE_COLORS.get(baseline_name, "gray")
        register(baseline_color)

        return color_defs, color_name_map

    n_plots = len(plot_names)
    hyper_values = list(exp_module.x_tick_values)
    hyper_ticklabels = list(exp_module.x_tick_labels)

    if n_plots == 0:
        return "\\begin{tikzpicture}\n\\end{tikzpicture}"

    if not hyper_values:
        raise ValueError("exp_module.x_tick_values is empty.")

    x_min = min(hyper_values)
    x_max = max(hyper_values)

    baseline_name, baseline_val = exp_module.baseline
    baseline_color_spec = BASELINE_COLORS.get(baseline_name, "gray")
    baseline_dash = BASELINE_DASHES[baseline_name]

    # Scalar baseline (old behavior) vs per-x baseline (new relative mode)
    baseline_is_series = isinstance(baseline_val, (tuple, list))

    if baseline_is_series:
        if len(baseline_val) != len(hyper_values):
            raise ValueError(
                "If baseline_val is a tuple/list, it must have exactly "
                f"len(hyper_values)={len(hyper_values)} entries, got {len(baseline_val)}."
            )
        baseline_scalar = None
        baseline_series = [float(v) for v in baseline_val]
    else:
        baseline_scalar = float(baseline_val)
        baseline_series = None

    color_defs, color_name_map = _collect_color_defs()

    def _color_name(color_spec):
        return color_name_map[repr(color_spec)]

    def _baseline_at_x(x):
        """
        Return the baseline value corresponding to x.
        Only used when baseline_is_series is True.
        """
        x = float(x)
        for hx, bv in zip(hyper_values, baseline_series):
            if math.isclose(x, float(hx), rel_tol=0.0, abs_tol=1e-12):
                return bv
        raise ValueError(
            f"x={x!r} occurs in run_matrix but is not present in exp_module.x_tick_values."
        )

    def _plot_options(color_spec, marker, dash, forget=False):
        opts = [
            f"color={_color_name(color_spec)}",
            "line width=1pt",
        ]

        pgf_marker = _marker_to_pgf(marker)
        if pgf_marker == "none":
            opts.append("mark=none")
        else:
            opts.append(f"mark={pgf_marker}")
            opts.append("mark size=2pt")
            opts.append("mark options={solid}")

        dash_opt = _normalize_dash_pattern(dash)
        if dash_opt is not None:
            opts.append(dash_opt)

        if forget:
            opts.append("forget plot")

        return ", ".join(opts)

    lines = []
    lines.append("% Requires:")
    lines.append("%   \\usepackage{pgfplots}")
    lines.append("%   \\usepgfplotslibrary{groupplots}")
    lines.append("%   \\pgfplotsset{compat=1.18}")
    lines.append("\\begin{tikzpicture}")

    for cdef in color_defs:
        lines.append(cdef)

    groupplot_header = [
        rf"\begin{{groupplot}}[",
        rf"  group style={{group size={n_plots} by 1, horizontal sep=0.7cm}},",
        r"  width=1.85in,",
        r"  height=1.5in,",
        r"  grid=major,",
        r"  grid style={opacity=0.3},",
        f"  tick label style={{font=\\footnotesize}},",
        r"]",
    ]
    lines.extend(groupplot_header)

    # Match original legend behavior: legend is derived from the first axis' plotted lines, plus baseline.
    for plot_idx in range(n_plots):
        axis_opts = [
            f"title={{{plot_names[plot_idx]}}}",
            f"title style={{align=center}}",
            f"xlabel={{{hyper_name}}}",
            f"xmin={_fmt_num(x_min)}",
            f"xmax={_fmt_num(x_max)}",
            "xtick={" + ",".join(_fmt_num(v) for v in hyper_values) + "}",
            "xticklabels={" + ",".join(hyper_ticklabels) + "}",
            f"xticklabel style={{align=center}}",
        ]

        if plot_idx == 0:
            ylabel = metric_name.split("/")[-1]
            if baseline_is_series:
                ylabel += f"\\\\vs. {BASELINE_LEGEND_NAMES[baseline_name]}"
            axis_opts.append(f"ylabel={{{ylabel}}}")
            if baseline_is_series:
                axis_opts.append(f"ylabel style={{align=center}}")
            axis_opts.append("legend to name=sharedlegend")
            axis_opts.append("legend columns=-1")
            axis_opts.append("legend style={draw=black, fill=white, font=\\footnotesize}")

        if hasattr(exp_module, "scales_boxplot"):
            vmin, vmax = exp_module.scales_boxplot
            axis_opts.append(f"ymin={_fmt_num(vmin)}")
            axis_opts.append(f"ymax={_fmt_num(vmax)}")

        lines.append(r"\nextgroupplot[" + ", ".join(axis_opts) + "]")

        line_info = zip(
            line_names,
            exp_module.line_colors,
            exp_module.line_markers,
            exp_module.line_dashes,
        )

        for line_idx, (line_name, line_color, line_marker, line_dash) in enumerate(line_info):
            key = (plot_idx, line_idx)
            if key not in run_matrix:
                continue

            xs, ys = run_matrix[key]
            if not xs:
                continue

            if ys and isinstance(ys[0], (list, tuple)):
                raise ValueError(
                    "create_tikz received unreduced replicate lists in ys. "
                    "Use reducer='mean' before plotting line plots."
                )

            if len(xs) != len(ys):
                raise ValueError(
                    f"Length mismatch for key {key}: len(xs)={len(xs)} != len(ys)={len(ys)}"
                )

            if baseline_is_series:
                ys_plot = []
                for x, y in zip(xs, ys):
                    b = _baseline_at_x(x)
                    if math.isclose(b, 0.0, rel_tol=0.0, abs_tol=1e-15):
                        raise ValueError(f"Baseline value is zero at x={x!r}; cannot normalize by zero.")
                    ys_plot.append(float(y) / b)
            else:
                ys_plot = ys

            coords = " ".join(
                f"({_fmt_num(x)},{_fmt_num(y)})"
                for x, y in zip(xs, ys_plot)
            )

            forget = (plot_idx != 0)
            opts = _plot_options(line_color, line_marker, line_dash, forget=forget)
            lines.append(rf"\addplot+[{opts}] coordinates {{{coords}}};")

            if plot_idx == 0:
                lines.append(rf"\addlegendentry{{{line_name}}}")

        # baseline
        baseline_forget = (plot_idx != 0)
        baseline_opts = [
            f"color={_color_name(baseline_color_spec)}",
            "line width=1pt",
            "mark=none",
        ]
        baseline_dash_opt = _normalize_dash_pattern(baseline_dash)
        if baseline_dash_opt is not None:
            baseline_opts.append(baseline_dash_opt)
        else:
            baseline_opts.append("dashed")

        if baseline_forget:
            baseline_opts.append("forget plot")

        if baseline_is_series:
            baseline_coords = " ".join(
                f"({_fmt_num(x)},{_fmt_num(1.0)})"
                for x in hyper_values
            )
        else:
            baseline_coords = (
                f"({_fmt_num(x_min)},{_fmt_num(baseline_scalar)}) "
                f"({_fmt_num(x_max)},{_fmt_num(baseline_scalar)})"
            )
        lines.append(rf"\addplot+[{', '.join(baseline_opts)}] coordinates {{{baseline_coords}}};")

        if plot_idx == 0:
            lines.append(rf"\addlegendentry{{{BASELINE_LEGEND_NAMES[baseline_name]}}}")

    lines.append(r"\end{groupplot}")

    # Place legend centered over the full tikzpicture bounding box.
    lines.append(
        r"\node[anchor=south] at ([yshift=0.1cm]current bounding box.north) "
        r"{\pgfplotslegendfromname{sharedlegend}};"
    )

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


def create_figure(run_matrix, plot_names, line_names, hyper_name, metric_name, exp_module, use_tikz=False):

    plotting_result_path = Path(DEFAULT_OUT_DIR) / "plotting_results"
    plotting_result_path.mkdir(exist_ok=True, parents=True)

    file_ending = 'tex' if use_tikz else 'png'
    filename = f"1metric_line_{matrix_config}_{datetime.now().strftime('%m%d-%H%M%S')}.{file_ending}"
    out_fp = str((plotting_result_path / filename).resolve())

    if use_tikz:
        print("Creating the tikz...")
        tikz_code = create_tikz_code(
            run_matrix, plot_names, line_names, hyper_name, metric_name, exp_module
        )
        with open(out_fp, "w") as f:
            f.write(tikz_code)
    else:
        print("Creating the plots...")
        fig = create_1metric_plots(
            run_matrix, plot_names, line_names, hyper_name, metric_name, exp_module
        )
        fig.subplots_adjust(wspace=0.04, hspace=0.04)
        fig.savefig(out_fp, format="png", dpi=300, bbox_inches="tight")


# =====================================================================================================================


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Crawl results from W&B and plot for a specific experiment.')
    parser.add_argument("-c", "--cached-dir", type=Path, default=None,
                        help="If set, load runs from this crawler cache directory instead of calling the W&B API.",
                        )
    parser.add_argument('--debug', '-d', action='store_true', help='Debug mode.')
    parser.add_argument('--matrix_config', '-m', type=str, required=True)
    parser.add_argument('--use_tikz', '-t', action='store_true', help='Output TikZ code instead of saving a PNG plot.')
    args = parser.parse_args()

    matrix_config = args.matrix_config
    exp_module = importlib.import_module(f"scripts.plot_config.{matrix_config}")

    # get results from wandb API
    all_runs = crawl_results(args, exp_module)

    # create run matrix
    run_matrix, plot_names, line_names, hyper_name, metric_name = get_run_matrix(all_runs, args, exp_module)

    # create figure
    create_figure(run_matrix, plot_names, line_names, hyper_name, metric_name, exp_module, args.use_tikz)
