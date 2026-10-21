from ._base import *

cols = [
    ("\\emph{LOGGIA}", lambda config: config["runs"]["algo"] == "ms"),

    ("\\emph{LOGGIA-path}", lambda config: config["runs"]["algo"] == "ms_path"),
    ("$+K_{\\text{rand}}$", lambda config: config["runs"]["algo"] == "ms_path_rand3"),
    ("$+K_{\\text{rand}};\\text{TRPL}$", lambda config: config["runs"]["algo"] == "ms_path_rand3_trpl"),
]

col_colors = [
    '#0173b2',  # blue
    '#ca9161',  # brown
    '#ca9161',  # brown
    '#ca9161',  # brown
]

col_hatches = [None, None, '/', 'x']

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
        (123, 243),
        (140, 340),
        (170, 470),
]

plots = [
        ("\\emph{mini5}", lambda config: config["runs"]["full_scenario"] == "m5_default"),
        ("\\emph{B4}", lambda config: config["runs"]["full_scenario"] == "b4_default"),
        ("\\emph{nx-XS}", lambda config: config["runs"]["full_scenario"] == "nx_xs_default"),
]

run_filters = {"config.runs.trainer": "il_ppo_c"}
exp_names = ["0217_ms_ablation"]