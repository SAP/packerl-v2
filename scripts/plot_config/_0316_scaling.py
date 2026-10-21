from ._base import *

def trained_without_obs_delay(config):
    return config["runs"]["trainer"].endswith("b")

def trained_with_obs_delay(config):
    return not config["runs"]["trainer"].endswith("b")

def trained_without_ac_delay(config):
    return config["train_inference_delay_scaling"] == 0

def trained_with_ac_delay(config):
    return config["train_inference_delay_scaling"] == 0.2

def is_ppo(config):
    return config["runs"]["trainer"].startswith("il_ppo")

def is_mappo(config):
    return config["runs"]["trainer"].startswith("il_mappo")

all_lines = [
        #("PPO, delay-oblivious",           lambda config: is_ppo(config) and trained_without_obs_delay(config) and trained_without_ac_delay(config)),
        ("PPO, delay-aware",               lambda config: is_ppo(config) and trained_with_obs_delay(config) and trained_with_ac_delay(config)),

        #("MAPPO, delay-oblivious",         lambda config: is_mappo(config) and trained_without_obs_delay(config) and trained_without_ac_delay(config)),
        ("MAPPO, delay-aware",             lambda config: is_mappo(config) and trained_with_obs_delay(config) and trained_with_ac_delay(config)),
]

line_colors = [
    '#56b4e9',  # light blue
    #'#56b4e9',  # light blue
    #'#4f67ab',  # dark blue
    '#4f67ab',  # dark blue
]

line_markers = [
    'o', 'v', 'o', 'v'
]

line_dashes  = [
    [3, 1], None, [3, 1], None
]

all_plots = [
        ("\\texttt{Birdseye-Single}\\\\deployment", lambda config: config["runs"]["eval_depl"] == "bs"),
        ("\\texttt{Central-Single}\\\\deployment", lambda config: config["runs"]["eval_depl"] == "cs"),
        ("\\texttt{Birdseye-Multi}\\\\deployment", lambda config: config["runs"]["eval_depl"] == "bm"),
        ("\\texttt{Local-Multi}\\\\deployment", lambda config: config["runs"]["eval_depl"] == "lm"),
]

x_tick_name = "Number of Nodes"
x_tick_values = [
    0,
    0.25,
    0.5,
    0.75,
    #1.0
]
x_scen_to_tick_values = {
    "nx_xs": 0,
    "nx_s": 0.25,
    "nx_m": 0.5,
    "nx_l": 0.75,
    "nx_xl": 1.0,
}
x_ticks = [(lambda config, v=v: x_scen_to_tick_values[config["runs"]["scenario"]] == v) for v in x_tick_values]
x_tick_labels = [
    "6\\\\-10",
    "11\\\\-25",
    "26\\\\-50",
    "51\\\\-100",
    #"101\\\\-250"
]

scales_boxplot = 0.95, 1.05
baseline = "rip", (
    443.72232,
    734.57473,
    1115.74016,
    1547.5632,
    #2493.15789
)

exp_names = ["0316_scaling"]
