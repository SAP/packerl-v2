from ._base import *

def is_right_loggia(config):
    return config["runs"]["algo"] == "ms" and (config["runs"]["scenario_mod"] == "default" and config["exp_name"] == "0219_trainer"
                                               or config["runs"]["scenario_mod"] != "default" and config["exp_name"] == "0310_champs") and config["runs"]["eval_depl"] == "lm"

cols = [
    ("\\emph{MAGNNETO}", lambda config: config["runs"]["algo"] == "magnneto"),
    ("\\emph{FieldLines}", lambda config: config["runs"]["algo"] == "fl_tmlr24" and config["runs"]["trainer"] == "il_ppo_c"),
    ("\\emph{M-Slim}", lambda config: config["runs"]["algo"] == "ms_tmlr24" and config["runs"]["trainer"] == "il_ppo_c"),
    ("\\emph{LOGGIA} (PPO)", lambda config: is_right_loggia(config) and config["runs"]["trainer"] == "il_ppo_c"),
    ("\\emph{LOGGIA} (MAPPO)", lambda config: is_right_loggia(config) and config["runs"]["trainer"] == "il_mappo_c"),
]

col_colors = [
    '#fbafe4',  # pink
    '#d55e00',  # red orange
    '#de8f05',  # orange
    '#56b4e9',  # light blue
    '#4f67ab',  # dark blue
]
col_hatches = [None] * len(col_colors)

plot_baselines = [
    ("rip", 391.7955),
    ("rip", 473.4882),
    ("rip", 442.7223),
    ("rip", 442.7223),
    ("rip", 442.7223),
]

legend_algo_labels = [c[0] for c in cols]
legend_algo_color_bucket_idx = range(len(cols))
legend_algo_hatches = col_hatches
legend_baseline_labels = list(set(b[0] for b in plot_baselines))

plot_limits = [
        (300, 410),
        (400, 500),
        (400, 465),
        (400, 465),
        (400, 465),
]

plots = [
    ("$50\%$ Buffer Size", lambda config: config["runs"]["full_scenario"] == "nx_xs_buffer_50p"),
    ("$200\%$ Buffer Size", lambda config: config["runs"]["full_scenario"] == "nx_xs_buffer_200p"),
    ("$\\tau = 10$ ms", lambda config: config["runs"]["full_scenario"] == "nx_xs_time_2s_10ms"),
    ("$\\tau = 5$ ms (default)", lambda config: config["runs"]["full_scenario"] == "nx_xs_default"),
    ("$\\tau = 2$ ms", lambda config: config["runs"]["full_scenario"] == "nx_xs_time_2s_2ms"),
]

exp_names = ["0310_champs", "0219_trainer"]
