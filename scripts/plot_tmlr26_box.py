import argparse
from collections import defaultdict
from datetime import datetime
import importlib
from pathlib import Path
import sys

sys.path.append(".")

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.cbook import boxplot_stats
import numpy as np

from src.utils.constants import DEFAULT_OUT_DIR
from scripts._plot_utils import crawl_results, BASELINE_LEGEND_NAMES, BASELINE_DASHES, BASELINE_COLORS


def create_boxplots_plt(grouped_data, plot_info, bucket_names, exp_module):
    num_plots = len(plot_info)
    num_buckets = len(bucket_names)
    plot_alignment = getattr(exp_module, "plot_alignment", "horizontal")
    col_colors = exp_module.col_colors
    col_hatches = exp_module.col_hatches

    populated_bucket_indices_per_plot = []
    width_ratios = []

    for plot_idx in range(num_plots):
        populated_bucket_indices = [
            bucket_idx
            for bucket_idx in range(num_buckets)
            if grouped_data.get((plot_idx, bucket_idx), [])
        ]
        populated_bucket_indices_per_plot.append(populated_bucket_indices)
        width_ratios.append(max(1, len(populated_bucket_indices)))

    # figure size
    if plot_alignment == "vertical":
        max_visible_buckets = max(width_ratios) if width_ratios else 1

        # width needed for the actual plot area
        plot_width_in = max(4.5, 0.55 * max_visible_buckets + 1.8)

        # reserve explicit space on the right for the legend
        legend_gap_in = 0.25
        legend_width_in = 2.2

        fig_width = plot_width_in + legend_gap_in + legend_width_in
        fig_height = max(2.2 * num_plots, 2.8)
    else:
        fig_width = max(1.8 * num_plots, 0.35 * sum(width_ratios) + 0.8 * num_plots)
        fig_height = 3.2

    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = []

    if plot_alignment == "vertical":
        # margins in figure coordinates
        left_margin = 0.10
        bottom_margin = 0.08
        top_margin = 0.95
        inter_ax_gap = 0.05

        # convert reserved legend space from inches to figure fractions
        legend_gap_frac = legend_gap_in / fig_width
        legend_width_frac = legend_width_in / fig_width

        # right margin includes the reserved legend strip
        right_margin = 0.03 + legend_gap_frac + legend_width_frac

        usable_height = (
            top_margin - bottom_margin - inter_ax_gap * (num_plots - 1)
        )
        ax_height = usable_height / max(1, num_plots)
        ax_width = 1.0 - left_margin - right_margin

        cur_bottom = top_margin - ax_height
        for _ in range(num_plots):
            ax = fig.add_axes([left_margin, cur_bottom, ax_width, ax_height])
            axes.append(ax)
            cur_bottom -= ax_height + inter_ax_gap

        # x-position where the legend should start
        legend_anchor_x = left_margin + ax_width + legend_gap_frac

    else:
        left_margin = 0.08
        right_margin = 0.02
        bottom_margin = 0.16
        top_margin = 0.80
        inter_ax_gap = 0.045

        usable_width = 1.0 - left_margin - right_margin - inter_ax_gap * (num_plots - 1)
        total_ratio = sum(width_ratios)

        cur_left = left_margin
        for ratio in width_ratios:
            ax_width = usable_width * (ratio / total_ratio)
            ax = fig.add_axes([cur_left, bottom_margin, ax_width, top_margin - bottom_margin])
            axes.append(ax)
            cur_left += ax_width + inter_ax_gap

    for plot_idx, ax in enumerate(axes):
        plotted_bucket_indices = populated_bucket_indices_per_plot[plot_idx]

        all_box_data = [
            grouped_data[(plot_idx, bucket_idx)]
            for bucket_idx in plotted_bucket_indices
        ]
        all_box_positions = np.arange(len(plotted_bucket_indices))

        if all_box_data:
            bp = ax.boxplot(
                all_box_data,
                positions=all_box_positions,
                widths=0.8,
                patch_artist=True,
                manage_ticks=False,
            )

            for box, bucket_idx in zip(bp["boxes"], plotted_bucket_indices):
                box.set_facecolor(col_colors[bucket_idx % len(col_colors)])
                box.set_alpha(0.7)
                box.set_hatch(col_hatches[bucket_idx % len(col_hatches)])

            for median in bp["medians"]:
                median.set_linewidth(1.5)

            ax.set_xlim(-0.5, len(plotted_bucket_indices) - 0.5)
        else:
            ax.set_xlim(-0.5, 0.5)

        plot_name, (ymin, ymax), (baseline_name, baseline_val) = plot_info[plot_idx]

        ax.set_ylim(bottom=ymin, top=ymax)

        baseline_color = BASELINE_COLORS.get(baseline_name, "gray")
        baseline_dash = BASELINE_DASHES[baseline_name]
        ax.axhline(
            y=baseline_val,
            color=baseline_color,
            linestyle="--",
            linewidth=2,
            dashes=baseline_dash,
        )

        ax.set_title(plot_name)

        ax.set_xticks([])
        ax.tick_params(axis="x", length=0)

        if plot_idx == 0 or plot_alignment == "vertical":
            ax.set_ylabel(
                exp_module.metric_name.split("/")[-1],
                fontsize=10 if plot_alignment == "vertical" else 12,
            )

    baseline_order = list(dict.fromkeys(baseline_name for _, _, (baseline_name, _) in plot_info))

    bucket_handles = [
        Patch(
            facecolor=col_colors[i % len(col_colors)],
            edgecolor="black",
            label=bucket_name,
            alpha=0.7,
            hatch=col_hatches[i % len(col_hatches)],
        )
        for i, bucket_name in enumerate(bucket_names)
    ]

    baseline_handles = [
        Line2D(
            [0],
            [0],
            color=BASELINE_COLORS.get(baseline_name, "gray"),
            linestyle="--",
            linewidth=2,
            label=BASELINE_LEGEND_NAMES[baseline_name],
            dashes=BASELINE_DASHES[baseline_name],
        )
        for baseline_name in baseline_order
    ]

    legend_handles = bucket_handles + baseline_handles

    if plot_alignment == "vertical":
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(legend_anchor_x, 0.5),
            bbox_transform=fig.transFigure,
            ncol=1,
            frameon=True,
        )
    else:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=max(1, len(legend_handles)),
            frameon=True,
        )

    return fig


def create_boxplots_tikz(
    grouped_data,
    plot_info,
    bucket_names,
    exp_module,
    *,
    axis_width_cm=None,
    axis_height_cm=None,
    inter_plot_gap_cm=None,
    scale_horizontal_widths_by_population=True,
    left_margin_cm=0.7,
    bottom_margin_cm=0.0,
    legend_gap_cm=0.0,
    legend_max_items_per_row=4,
    show_confidence_interval=False,
):
    import math
    import numpy as np
    from matplotlib import colors as mcolors
    from matplotlib.cbook import boxplot_stats
    from matplotlib.ticker import MaxNLocator

    def _fmt_num(x):
        x = float(x)
        if not math.isfinite(x):
            raise ValueError(f"Non-finite value encountered: {x!r}")
        return f"{x:.12g}"

    def _fmt_cm(x):
        x = float(x)
        if not math.isfinite(x):
            raise ValueError(f"Non-finite dimension encountered: {x!r}")
        return f"{x:.6f}"

    def _has_data(values):
        if values is None:
            return False
        arr = np.asarray(values)
        return arr.size > 0

    def _flatten_numeric(values):
        arr = np.asarray(values, dtype=float).ravel()
        return arr[np.isfinite(arr)]

    def _mpl_hatch_to_tikz(hatch):
        if not hatch:
            return None
        for ch in hatch:
            if ch in "/\\|-+x.oO*5":
                return {
                    "/": "north east lines",
                    "\\": "north west lines",
                    "|": "vertical lines",
                    "-": "horizontal lines",
                    "+": "grid",
                    "x": "crosshatch",
                    ".": "dots",
                    "o": "dots",
                    "O": "crosshatch dots",
                    "*": "crosshatch dots",
                    "5": "fivepointed stars",
                }.get(ch)
        return None

    def _normalize_dash_pattern(dash):
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

            if len(seq) == 2 and isinstance(seq[1], (tuple, list)):
                seq = list(seq[1])

            if not seq:
                return None

            seq = [float(v) for v in seq if float(v) > 0]
            if not seq:
                return None

            if len(seq) == 2 and seq[1] == 0:
                return None

            if len(seq) % 2 == 1:
                seq = seq * 2

            parts = []
            for i, v in enumerate(seq):
                parts.append(("on " if i % 2 == 0 else "off ") + f"{_fmt_num(v)}pt")
            return "dash pattern=" + " ".join(parts)

        return None

    def _solid_dash_pattern():
        return "dash pattern="

    def _dash_or_solid(dash):
        dash_opt = _normalize_dash_pattern(dash)
        return dash_opt if dash_opt is not None else _solid_dash_pattern()

    def _define_rgb_color(name, mpl_color):
        r, g, b, _ = mcolors.to_rgba(mpl_color)
        return (
            rf"\definecolor{{{name}}}{{RGB}}"
            rf"{{{int(round(255 * r))},{int(round(255 * g))},{int(round(255 * b))}}}"
        )

    def _legend_patch_tikz(color_name, hatch):
        style = [
            "draw=black",
            f"fill={color_name}",
            "fill opacity=0.7",
            "line width=0.5pt",
            _solid_dash_pattern(),
        ]
        mapped = _mpl_hatch_to_tikz(hatch)
        if mapped:
            style += [f"pattern={mapped}", f"pattern color={color_name}"]
        return r"\tikz{\draw[" + ", ".join(style) + r"] (0,0) rectangle (0.34cm,0.18cm);}"

    def _legend_line_tikz(color_name, dash):
        style = [
            f"draw={color_name}",
            "line width=1.4pt",
            _dash_or_solid(dash),
        ]
        return r"\tikz{\draw[" + ", ".join(style) + r"] (0,0.09cm) -- (0.34cm,0.09cm);}"

    def _normalize_y_spec(y_spec):
        y_spec = tuple(float(v) for v in y_spec)
        if len(y_spec) == 2:
            ymin, ymax = y_spec
            if not ymin < ymax:
                raise ValueError(f"Expected ymin < ymax, got {y_spec!r}")
            return {
                "broken": False,
                "raw_min": ymin,
                "raw_max": ymax,
            }
        if len(y_spec) == 3:
            ymin, ybreak, ymax = y_spec
            if not (ymin < ybreak < ymax):
                raise ValueError(f"Expected ymin < ybreak < ymax, got {y_spec!r}")
            return {
                "broken": True,
                "raw_min": ymin,
                "raw_break": ybreak,
                "raw_max": ymax,
            }
        raise ValueError(
            f"y-limits must be a 2-tuple (ymin,ymax) or 3-tuple (ymin,ybreak,ymax), got {y_spec!r}"
        )

    def _build_broken_yticks(y_cfg, lower_bins=4, upper_bins=4):
        ymin = y_cfg["raw_min"]
        ybreak = y_cfg["raw_break"]
        ymax = y_cfg["raw_max"]

        lower_locator = MaxNLocator(nbins=lower_bins)
        upper_locator = MaxNLocator(nbins=upper_bins)

        lower_ticks = lower_locator.tick_values(ymin, ybreak)
        upper_ticks = upper_locator.tick_values(ybreak, ymax)

        tick_vals = []
        seen = set()

        for t in np.concatenate([lower_ticks, upper_ticks]):
            t = float(t)
            if ymin - 1e-12 <= t <= ymax + 1e-12:
                key = round(t, 12)
                if key not in seen:
                    seen.add(key)
                    tick_vals.append(t)

        tick_vals.sort()
        return tick_vals

    def _zigzag_rel_axis_path(n_zigs=18, amp=0.012):
        pts = ["(rel axis cs:0,0.5)"]
        for i in range(1, 2 * n_zigs):
            x = i / (2 * n_zigs)
            y = 0.5 + (amp if i % 2 else -amp)
            pts.append(f"(rel axis cs:{_fmt_num(x)},{_fmt_num(y)})")
        pts.append("(rel axis cs:1,0.5)")
        return " -- ".join(pts)

    num_plots = len(plot_info)
    num_buckets = len(bucket_names)

    if num_plots == 0:
        return "\\begin{tikzpicture}\\end{tikzpicture}"

    plot_alignment = getattr(exp_module, "plot_alignment", "horizontal")
    col_colors = exp_module.col_colors
    col_hatches = exp_module.col_hatches

    populated_bucket_indices_per_plot = []
    width_ratios = []

    for plot_idx in range(num_plots):
        populated_bucket_indices = [
            bucket_idx
            for bucket_idx in range(num_buckets)
            if _has_data(grouped_data.get((plot_idx, bucket_idx), []))
        ]
        populated_bucket_indices_per_plot.append(populated_bucket_indices)
        width_ratios.append(max(1, len(populated_bucket_indices)))

    # Defaults chosen to stay close to the earlier behavior.
    if plot_alignment == "vertical":
        max_visible_buckets = max(width_ratios) if width_ratios else 1
        if axis_width_cm is None:
            axis_width_cm = max(4.5, 0.55 * max_visible_buckets + 1.8) * 2.54
        if axis_height_cm is None:
            axis_height_cm = 2.2 * 2.54
        if inter_plot_gap_cm is None:
            inter_plot_gap_cm = 0.22
    else:
        # In horizontal mode:
        # - if scale_horizontal_widths_by_population=True, axis_width_cm is the width per ratio unit
        # - otherwise, axis_width_cm is the exact width of every axis
        if axis_width_cm is None:
            axis_width_cm = 1.35
        if axis_height_cm is None:
            axis_height_cm = 5.20
        if inter_plot_gap_cm is None:
            inter_plot_gap_cm = 0.25

    lines = []
    lines.append("% Requires:")
    lines.append("%   \\usepackage{pgfplots}")
    lines.append("%   \\usetikzlibrary{patterns,matrix}")
    lines.append("%   \\usepgfplotslibrary{statistics}")
    lines.append("%   \\pgfplotsset{compat=1.18}")
    lines.append(r"\providecommand{\isPGFNumber}[3]{%")
    lines.append(r"  \begingroup")
    lines.append(r"  \pgfkeys{/pgf/fpu/handlers/invalid number/.code={}}%")
    lines.append(r"  \pgfmathfloatparsenumber{#1}%")
    lines.append(r"  \global\pgfmathfloatgetflagstomacro\pgfmathresult\pgfFloatParseFlag%")
    lines.append(r"  \endgroup")
    lines.append(r"  \ifnum\pgfFloatParseFlag=3 #3\else #2\fi%")
    lines.append(r"}")
    lines.append(r"\begin{tikzpicture}")

    bucket_color_names = []
    for i in range(num_buckets):
        color_name = f"boxplotBucketColor{i}"
        bucket_color_names.append(color_name)
        lines.append(_define_rgb_color(color_name, col_colors[i % len(col_colors)]))
    bucket_color_names.append("black")  # default
    bucket_color_names.append("white")  # default

    baseline_order = list(
        dict.fromkeys(baseline_name for _, _, (baseline_name, _) in plot_info)
    )
    baseline_color_names = {}
    for i, baseline_name in enumerate(baseline_order):
        color_name = f"boxplotBaselineColor{i}"
        baseline_color_names[baseline_name] = color_name
        lines.append(
            _define_rgb_color(color_name, BASELINE_COLORS.get(baseline_name, "gray"))
        )

    axes_geometry = []

    if plot_alignment == "vertical":
        total_axes_height_cm = (
            num_plots * axis_height_cm + max(0, num_plots - 1) * inter_plot_gap_cm
        )

        for plot_idx in range(num_plots):
            bottom = (
                bottom_margin_cm
                + total_axes_height_cm
                - (plot_idx + 1) * axis_height_cm
                - plot_idx * inter_plot_gap_cm
            )
            axes_geometry.append(
                {
                    "left": left_margin_cm,
                    "bottom": bottom,
                    "width": axis_width_cm,
                    "height": axis_height_cm,
                }
            )
    else:
        cur_left = left_margin_cm
        for ratio in width_ratios:
            width_cm = axis_width_cm * ratio if scale_horizontal_widths_by_population else axis_width_cm
            axes_geometry.append(
                {
                    "left": cur_left,
                    "bottom": bottom_margin_cm,
                    "width": width_cm,
                    "height": axis_height_cm,
                }
            )
            cur_left += width_cm + inter_plot_gap_cm

    ylabel_text = exp_module.metric_name.split("/")[-1]

    for plot_idx, geom in enumerate(axes_geometry):
        plotted_bucket_indices = populated_bucket_indices_per_plot[plot_idx]
        plot_name, y_spec, (baseline_name, baseline_val) = plot_info[plot_idx]
        y_cfg = _normalize_y_spec(y_spec)

        if plotted_bucket_indices:
            xmin = -0.5
            xmax = len(plotted_bucket_indices) - 0.5
        else:
            xmin, xmax = -0.5, 0.5

        axis_options = [
            rf"at={{({_fmt_cm(geom['left'])}cm,{_fmt_cm(geom['bottom'])}cm)}}",
            "anchor=south west",
            rf"width={_fmt_cm(geom['width'])}cm",
            rf"height={_fmt_cm(geom['height'])}cm",
            rf"xmin={_fmt_num(xmin)}",
            rf"xmax={_fmt_num(xmax)}",
            rf"ymin={_fmt_num(y_cfg['raw_min'])}",
            rf"ymax={_fmt_num(y_cfg['raw_max'])}",
            "xtick=\\empty",
            "ytick pos=left",
            "tick align=outside",
            "axis lines=box",
            "boxplot/draw direction=y",
            "clip=true",
            "tick label style={font=\\footnotesize}",
            rf"title={{{plot_name}}}",
        ]

        if y_cfg["broken"]:
            ymin_s = _fmt_num(y_cfg["raw_min"])
            ybreak_s = _fmt_num(y_cfg["raw_break"])
            ymax_s = _fmt_num(y_cfg["raw_max"])

            y_coord_trafo = (
                r"y coord trafo/.code={"
                r"\isPGFNumber{#1}{"
                rf"\pgfmathparse{{(#1 <= {ybreak_s} ? 0.5*((#1)-({ymin_s}))/(({ybreak_s})-({ymin_s})) : 0.5 + 0.5*((#1)-({ybreak_s}))/(({ymax_s})-({ybreak_s}))}}"
                r"}{}"
                r"}"
            )
            y_coord_inv_trafo = (
                r"y coord inv trafo/.code={"
                rf"\pgfmathparse{{(#1 <= 0.5 ? ({ymin_s}) + 2*(#1)*(({ybreak_s})-({ymin_s})) : ({ybreak_s}) + 2*((#1)-0.5)*(({ymax_s})-({ybreak_s}))}}"
                r"}"
            )

            axis_options.append(y_coord_trafo)
            axis_options.append(y_coord_inv_trafo)

            raw_ticks = _build_broken_yticks(y_cfg)
            axis_options.append(
                "ytick={" + ",".join(_fmt_num(v) for v in raw_ticks) + "}"
            )

        if plot_idx == 0 or plot_alignment == "vertical":
            axis_options.append(rf"ylabel={{{ylabel_text}}}")

        lines.append(r"\begin{axis}[")
        lines.extend("    " + opt + "," for opt in axis_options[:-1])
        lines.append("    " + axis_options[-1])
        lines.append(r"]")

        for draw_pos, bucket_idx in enumerate(plotted_bucket_indices):
            values = _flatten_numeric(grouped_data[(plot_idx, bucket_idx)])
            if values.size == 0:
                continue

            if show_confidence_interval:
                stats = boxplot_stats(values, whis=1.5, bootstrap=10000)[0]
            else:
                stats = boxplot_stats(values, whis=1.5)[0]

            style_parts = [
                "draw=black",
                rf"fill={bucket_color_names[bucket_idx]}",
                "fill opacity=0.7",
                "line width=0.8pt",
                _solid_dash_pattern(),
                rf"boxplot/draw position={_fmt_num(draw_pos)}",
                "boxplot/box extend=0.64",
                (
                    "boxplot prepared={"
                    + rf"lower whisker={_fmt_num(stats['whislo'])}, "
                    + rf"lower quartile={_fmt_num(stats['q1'])}, "
                    + rf"median={_fmt_num(stats['med'])}, "
                    + rf"upper quartile={_fmt_num(stats['q3'])}, "
                    + rf"upper whisker={_fmt_num(stats['whishi'])}"
                    + "}"
                ),
                rf"boxplot/every box/.style={{dash pattern={_solid_dash_pattern().split('=',1)[1]}}}",
                rf"boxplot/every whisker/.style={{dash pattern={_solid_dash_pattern().split('=',1)[1]}}}",
                rf"boxplot/every median/.style={{line width=1.5pt,dash pattern={_solid_dash_pattern().split('=',1)[1]}}}",
            ]

            hatch = _mpl_hatch_to_tikz(col_hatches[bucket_idx % len(col_hatches)])
            if hatch:
                style_parts.append(rf"pattern={hatch}")
                style_parts.append(f"pattern color={bucket_color_names[bucket_idx]}")

            if show_confidence_interval and 'cilo' in stats and 'cihi' in stats:
                # Add a semi-transparent dark ribbon representing the CI behind the box
                lines.append(
                    r"\fill[black, fill opacity=0.25, draw=none] "
                    f"({_fmt_num(draw_pos-0.2)},{_fmt_num(stats['cilo'])}) rectangle ({_fmt_num(draw_pos+0.2)},{_fmt_num(stats['cihi'])});"
                )

            lines.append(
                r"\addplot+["
                + ", ".join(style_parts)
                + "] coordinates {};"
            )

            fliers = np.asarray(stats["fliers"], dtype=float).ravel()
            if fliers.size:
                coords = " ".join(
                    f"({_fmt_num(draw_pos)},{_fmt_num(y)})" for y in fliers
                )
                lines.append(
                    r"\addplot+[only marks, mark=o, mark size=1.2pt, "
                    r"mark options={solid, fill=white, draw=black}, "
                    + _solid_dash_pattern()
                    + "] "
                    + f"coordinates {{{coords}}};"
                )

        baseline_opts = [
            "no markers",
            rf"color={baseline_color_names[baseline_name]}",
            "line width=2pt",
            _dash_or_solid(BASELINE_DASHES[baseline_name]),
        ]

        lines.append(
            r"\addplot+["
            + ", ".join(baseline_opts)
            + "] coordinates "
            + f"{{({_fmt_num(xmin)},{_fmt_num(baseline_val)}) ({_fmt_num(xmax)},{_fmt_num(baseline_val)})}};"
        )

        if y_cfg["broken"]:
            lines.append(
                r"\draw[draw=black!40, line width=0.35pt, opacity=0.6] "
                + _zigzag_rel_axis_path()
                + r";"
            )

        lines.append(r"\end{axis}")

    axes_left = min(g["left"] for g in axes_geometry)
    axes_right = max(g["left"] + g["width"] for g in axes_geometry)
    axes_top = max(g["bottom"] + g["height"] for g in axes_geometry)

    legend_items = []

    legend_patch_info = zip(exp_module.legend_algo_labels,
                            exp_module.legend_algo_color_bucket_idx,
                            exp_module.legend_algo_hatches)
    for p_label, p_color_bucket_idx, p_hatch in legend_patch_info:
        legend_patch = _legend_patch_tikz(bucket_color_names[p_color_bucket_idx], p_hatch)
        legend_items.append((legend_patch, p_label))

    for baseline_label in exp_module.legend_baseline_labels:
        legend_line = _legend_line_tikz(baseline_color_names[baseline_label], BASELINE_DASHES[baseline_label])
        legend_items.append((legend_line, BASELINE_LEGEND_NAMES[baseline_label]))

    if legend_max_items_per_row is None:
        legend_max_items_per_row = 6 if plot_alignment == "vertical" else 4

    n_rows = max(
        1, (len(legend_items) + legend_max_items_per_row - 1) // legend_max_items_per_row
    )

    legend_rows = []
    for row_idx in range(n_rows):
        row_items = legend_items[
            row_idx * legend_max_items_per_row : (row_idx + 1) * legend_max_items_per_row
        ]
        row_cells = []
        for symbol_tex, label_tex in row_items:
            row_cells.append(symbol_tex)
            row_cells.append("{" + str(label_tex) + "}")
        legend_rows.append(" & ".join(row_cells) + r" \\")

    legend_x_cm = 0.5 * (axes_left + axes_right)
    legend_y_cm = axes_top + legend_gap_cm

    lines.append(
        r"\matrix["
        r"matrix of nodes,"
        r"draw=black,"
        r"fill=white,"
        r"rounded corners=1pt,"
        r"inner sep=2pt,"
        r"outer sep=0pt,"
        r"anchor=south,"
        r"row sep=1.2mm,"
        r"column sep=1.8mm,"
        f"font=\\footnotesize,"
        r"nodes={anchor=west, inner sep=0pt, outer sep=0pt, text depth=0.15ex, text height=1.6ex}"
        rf"] at ({_fmt_cm(legend_x_cm)}cm,{_fmt_cm(legend_y_cm)}cm) {{"
    )
    for row in legend_rows:
        lines.append("  " + row)
    lines.append(r"};")

    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def get_run_buckets(all_runs, args, exp_module):
    print("Preparing buckets...")
    metric_name = exp_module.metric_name

    buckets = exp_module.cols
    bucket_names = [n for n, _ in buckets]
    bucket_filters = [f for _, f in buckets]

    plots = exp_module.plots
    plot_limits = exp_module.plot_limits
    plot_baselines = exp_module.plot_baselines

    plot_info = []
    plot_filters = []
    for i, (n, f) in enumerate(plots):
        plot_info.append((n, plot_limits[i], plot_baselines[i]))
        plot_filters.append(f)

    grouped = defaultdict(list)

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

        bucket_indices = [i for i, f in enumerate(bucket_filters) if f(cfg)]
        plot_idx = next((i for i, f in enumerate(plot_filters) if f(cfg)), None)

        if not bucket_indices or plot_idx is None:
            if args.debug:
                print(f"Run {r.id} does not match any bucket or plot filter.")
            continue

        for bucket_idx in bucket_indices:
            grouped[(plot_idx, bucket_idx)].append(metric)

    return grouped, plot_info, bucket_names


def create_figure(data_buckets, plot_info, bucket_names, exp_module, matrix_config, use_tikz, show_confidence_interval=False):

    plotting_result_path = Path(DEFAULT_OUT_DIR) / "plotting_results"
    plotting_result_path.mkdir(exist_ok=True, parents=True)

    file_ending = 'tex' if use_tikz else 'png'
    filename = f"tmlr26_box_{matrix_config}_{datetime.now().strftime('%m%d-%H%M%S')}.{file_ending}"
    out_fp = str((plotting_result_path / filename).resolve())

    if use_tikz:
        print("Creating the tikz...")
        tikz_code = create_boxplots_tikz(
            data_buckets, plot_info, bucket_names, exp_module,
            show_confidence_interval=show_confidence_interval
        )
        with open(out_fp, "w") as f:
            f.write(tikz_code)
    else:
        print("Creating the plots...")
        fig = create_boxplots_plt(data_buckets, plot_info, bucket_names, exp_module)
        fig.savefig(out_fp, format="png", dpi=300, bbox_inches="tight")


# =====================================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl results from W&B and plot for a specific experiment."
    )
    parser.add_argument(
        "-c",
        "--cached-dir",
        type=Path,
        default=None,
        help="If set, load runs from this crawler cache directory instead of calling the W&B API.",
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode.")
    parser.add_argument("--matrix_config", "-m", type=str, required=True)
    parser.add_argument('--use_tikz', '-t', action='store_true', help='Output TikZ code instead of saving a PNG plot.')
    parser.add_argument('--show-confidence-interval', '-i', action='store_true', help='Show bootstrapped 95%% confidence intervals.')
    args = parser.parse_args()

    matrix_config = args.matrix_config
    exp_module = importlib.import_module(f"scripts.plot_config.{matrix_config}")

    # get results from wandb API
    all_runs = crawl_results(args, exp_module)

    # create run matrix
    grouped_data, plot_info, bucket_names = get_run_buckets(all_runs, args, exp_module)

    # create boxplot figure
    create_figure(grouped_data, plot_info, bucket_names, exp_module, matrix_config, args.use_tikz, args.show_confidence_interval)