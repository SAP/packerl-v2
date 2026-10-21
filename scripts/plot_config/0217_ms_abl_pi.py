from ._base import *

cols = [
    ("\\emph{LOGGIA}", lambda config: config["runs"]["algo"] == "ms"),
    ("Log space", lambda config: config["runs"]["algo"] == "ms_flat_space"),
    ("Line graph", lambda config: config["runs"]["algo"] == "ms_line_graph"),
    ("$L=2, d_{\\text{GNN}}=12$", lambda config: config["runs"]["algo"] == "ms_small_nn"),

    ("\\emph{M-Slim}", lambda config: config["runs"]["algo"] == "ms_tmlr24"),
    ("Flat space", lambda config: config["runs"]["algo"] == "ms_tmlr24_log_space"),
    ("Base graph", lambda config: config["runs"]["algo"] == "ms_tmlr24_base_graph"),
    ("$L=4, d_{\\text{GNN}}=32$", lambda config: config["runs"]["algo"] == "ms_tmlr24_big_nn"),
]

col_colors = [
    '#0173b2',  # blue
    '#0173b2',  # blue
    '#0173b2',  # blue
    '#0173b2',  # blue
    '#de8f05',  # orange
    '#de8f05',  # orange
    '#de8f05',  # orange
    '#de8f05',  # orange
]

col_hatches = [None, '/', '\\', '.'] * 2

plot_baselines = [
    ("rip", 237.5695),
    ("rip", 329.4728),
    ("eigrp", 442.7223),
]

legend_algo_labels = [c[0] for c in cols]
legend_algo_color_bucket_idx = range(len(cols))
legend_algo_hatches = col_hatches
legend_baseline_labels = list(set(b[0] for b in plot_baselines))

plot_limits = [
        (193, 243),
        (280, 340),
        (350, 470),
]

plots = [
        ("\\emph{mini5}", lambda config: config["runs"]["full_scenario"] == "m5_default"),
        ("\\emph{B4}", lambda config: config["runs"]["full_scenario"] == "b4_default"),
        ("\\emph{nx-XS}", lambda config: config["runs"]["full_scenario"] == "nx_xs_default"),
]

run_filters = {"config.runs.trainer": "il_ppo_c"}
exp_names = ["0217_ms_ablation"]