import argparse
from datetime import datetime
from copy import deepcopy
import importlib
from pathlib import Path
import sys

from collections import defaultdict

sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.constants import DEFAULT_OUT_DIR
from scripts._plot_utils import crawl_results


def create_mat(run_grid, matrix_names, all_rows, col_names, baselines):
    """
    Create one LaTeX table (booktabs style) per experiment and return them
    as a list of strings.

    Layout is transposed relative to the original heatmap version:
      - rows    -> baseline + algorithms
      - columns -> metrics

    Parameters
    ----------
    run_grid : list[list[list[dict]]]
        run_grid[i][k] is a list of metric dictionaries for L runs of algorithm k
        on experiment i.
    matrix_names : list[str]
        Titles for the N experiments. Used as captions.
    all_rows : list[tuple[str, str]]
        List of metric definitions: (display_name, key_name).
        These become the columns of each table.
    col_names : list[str]
        Names of the K algorithm rows.
    baselines : dict | list | tuple
        Per-experiment baseline specification. Supported forms:
          - dict keyed by experiment index: baselines[i] = (baseline_name, baseline_metrics)
          - dict keyed by experiment name: baselines[matrix_names[i]] = (...)
          - list/tuple of length N with entries (baseline_name, baseline_metrics)

        baseline_metrics is a single dict of metric values.

    Returns
    -------
    list[str]
        One LaTeX table string per experiment.
    """

    def _to_float(x):
        try:
            if x is None:
                return np.nan
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    def _fmt_value(x):
        try:
            if x is None or np.isnan(x):
                return "N/A"
        except TypeError:
            return "N/A"
        return f"{x:.3f}"

    def _mean_metric(run_dicts, metric_key):
        if not run_dicts:
            return np.nan

        vals = []
        for d in run_dicts:
            if not isinstance(d, dict):
                continue
            v = _to_float(d.get(metric_key, np.nan))
            if not np.isnan(v):
                vals.append(v)

        return np.mean(vals) if vals else np.nan

    def _baseline_for(exp_idx, exp_name):
        entry = None
        if isinstance(baselines, dict):
            if exp_idx in baselines:
                entry = baselines[exp_idx]
            elif exp_name in baselines:
                entry = baselines[exp_name]
        elif isinstance(baselines, (list, tuple)) and exp_idx < len(baselines):
            entry = baselines[exp_idx]

        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and isinstance(entry[1], dict)
        ):
            return entry[0], entry[1]

        return "Baseline", {}

    def _latex_escape(s):
        s = str(s)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for old, new in replacements.items():
            s = s.replace(old, new)
        return s

    latex_tables = []

    n_metrics = len(all_rows)

    for i, exp_name in enumerate(matrix_names):
        baseline_name, baseline_metrics = _baseline_for(i, exp_name)

        row_labels = [baseline_name] + list(col_names)
        col_labels = [display_name for display_name, _ in all_rows]

        # Keep both numeric values and displayed cell text, matching the old logic.
        data = np.full((len(row_labels), n_metrics), np.nan, dtype=float)
        annot = np.empty((len(row_labels), n_metrics), dtype=object)

        # First row: baseline
        for c, (_, metric_key) in enumerate(all_rows):
            val = _to_float(baseline_metrics.get(metric_key, np.nan))
            data[0, c] = val
            annot[0, c] = _fmt_value(val)

        # Remaining rows: algorithms
        for k, _algo_name in enumerate(col_names):
            runs = []
            if i < len(run_grid) and k < len(run_grid[i]):
                runs = run_grid[i][k]

            for c, (_, metric_key) in enumerate(all_rows):
                mean_val = _mean_metric(runs, metric_key)
                data[k + 1, c] = mean_val
                annot[k + 1, c] = _fmt_value(mean_val)

        # Build DataFrame from displayed cell text.
        df = pd.DataFrame(annot, index=row_labels, columns=col_labels)

        tabular = df.to_latex(
            index=True,
            escape=True,
            index_names=False,
            bold_rows=False,
            column_format="l" + "c" * len(col_labels),
        )

        latex_table = "\n".join(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                f"\\caption{{{_latex_escape(exp_name)}}}",
                tabular,
                r"\end{table}",
            ]
        )

        latex_tables.append((exp_name, latex_table))

    return latex_tables


def get_run_dd(all_runs, args: argparse.Namespace, exp_module):

    all_plots = exp_module.all_plots
    all_rows = exp_module.all_rows
    all_cols = exp_module.all_cols

    plot_names = [name for name, _ in all_plots]
    col_names = [name for name, _ in all_cols]

    # defaultdict 'grouped' with key (plot_idx, row_idx, col_idx) -> [metrics, ...]
    grouped = defaultdict(list)

    primary_metric = exp_module.primary_metric
    for r in all_runs:
        # filter out runs that did not finish
        if primary_metric not in r.summary:
            if args.debug:
                print(f"Run {r.id} did not finish.")
            continue

        cfg = dict(r.config)
        summary = r.summary
        metrics = dict()
        for _, m in all_rows:
            metric = summary.get(m, None)
            if metric is None:
                if args.debug:
                    print(f"Run {r.id} does not have metric {m} in summary.")
                continue
            metrics[m] = metric

        plot_idxs = [i for i, (_, f) in enumerate(all_plots) if f(cfg)]
        col_idxs = [i for  i, (_, f) in enumerate(all_cols) if f(cfg)]

        if not plot_idxs or not col_idxs:
            if args.debug:
                print(
                    f"Run {r.id} did not match plot/col criteria: "
                    f"plot_idxs={plot_idxs}, col_idxs={col_idxs}"
                )
            continue

        for plot_idx in plot_idxs:
            for col_idx in col_idxs:
                grouped[(plot_idx, col_idx)].append(metrics)

    return grouped, plot_names, all_rows, col_names


def get_filter_func(seed_range_mode: str):
    """
    Returns a function that filters the runs based on the seed_range mode.
    Runs are sorted in ascending order of the metric value.
    """
    if seed_range_mode == 'worst':
        return lambda runs: runs[:1]
    elif seed_range_mode == 'lower25':  # include the first 25% of runs (at least one run)
        return lambda runs: runs[:len(runs) // 4] if len(runs) >= 4 else runs[:1]
    elif seed_range_mode == 'iqm':  # include the middle 50% of runs (at least one run)
        return lambda runs: runs[len(runs) // 4:-len(runs) // 4] if len(runs) >= 4 \
            else runs[1:2] if len(runs) == 3 \
            else runs
    elif seed_range_mode == 'upper50':  # include the best 50% of runs (at least one run)
        return lambda runs: runs[len(runs) // 2:] if len(runs) >= 2 else runs[-1:]
    elif seed_range_mode == 'upper25':  # include the last 25% of runs (at least one run)
        return lambda runs: runs[-len(runs) // 4:] if len(runs) >= 4 else runs[-1:]
    elif seed_range_mode == 'median':
        return lambda runs: runs[len(runs) // 2:len(runs) // 2 + 1] if len(runs) >= 3 else runs[1:2] if len(runs) == 3 else runs
    elif seed_range_mode == 'best':
        return lambda runs: runs[-1:]
    elif seed_range_mode == 'all':
        return lambda runs: runs
    else:
        raise ValueError(f"Invalid include mode: {seed_range_mode}")


def create_tables(run_grid, matrix_names, all_rows, col_names, matrix_config,
                  exp_module, seed_range, debug):
    # create the plot
    print("Creating the tables...")
    tables = create_mat(run_grid, matrix_names, all_rows, col_names, exp_module.baselines)

    plotting_result_path = Path(DEFAULT_OUT_DIR) / "plotting_results"
    plotting_result_path.mkdir(exist_ok=True, parents=True)

    for table_name, table_content in tables:
        filename = f"champs_table_{table_name}_{matrix_config}_{datetime.now().strftime('%m%d-%H%M%S')}_{seed_range}.tex"
        with open(str((plotting_result_path / filename).resolve()), "w") as f:
            f.write(table_content)


# =====================================================================================================================


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Crawl results from W&B and plot for a specific experiment.')
    parser.add_argument(
        "-c",
        "--cached-dir",
        type=Path,
        default=None,
        help="If set, load runs from this crawler cache directory instead of calling the W&B API.",
    )
    parser.add_argument('--debug', '-d', action='store_true', help='Debug mode.')
    parser.add_argument('--seed_range', '-s', type=str,
                        help='For methods with multiple random seeds, '
                             'Which seeds to include in the plot (with respect to the obtained episode reward). '
                             'Options: [all, iqm (runs that fall within the quartiles), '
                             'upper25 (the best 25% of runs), best].',
                        choices=['worst', 'lower25', 'iqm', 'upper50', 'upper25', 'median', 'best', 'all'],
                        nargs='+', default='iqm')
    parser.add_argument('--matrix_config', '-m', type=str, required=True)
    args = parser.parse_args()

    matrix_config = args.matrix_config
    exp_module = importlib.import_module(f"scripts.plot_config.{matrix_config}")

    # get results from wandb API
    all_runs = crawl_results(args, exp_module)

    # create run matrix
    run_dd, matrix_names, all_rows, col_names = get_run_dd(all_runs, args, exp_module)
    primary_metric = exp_module.primary_metric

    seed_ranges = args.seed_range
    if isinstance(seed_ranges, str):
        seed_ranges = [seed_ranges]

    for seed_range in seed_ranges:
        run_grid = [[[] for _ in range(len(col_names))] for _ in range(len(matrix_names))]
        filter_func = get_filter_func(seed_range)
        for i in range(len(matrix_names)):
            for k in range(len(col_names)):
                cell_runs = sorted(run_dd[(i, k)], key=lambda x: x[primary_metric])
                filtered_runs = filter_func(cell_runs)
                run_grid[i][k] = filtered_runs
                if args.debug:
                    print(f"After filtering Cell ({i}, {k}): {len(filtered_runs)} runs remain.")
        create_tables(run_grid, matrix_names, all_rows, col_names, matrix_config, exp_module, seed_range, args.debug)
