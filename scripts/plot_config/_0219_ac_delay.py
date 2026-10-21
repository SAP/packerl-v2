from ._base import *

def is_magnneto(config):
    return config["runs"]["algo"] == "magnneto"

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
        ("PPO, delay-oblivious",           lambda config: is_ppo(config) and trained_without_obs_delay(config) and trained_without_ac_delay(config)),
        ("PPO, delay-aware",               lambda config: is_ppo(config) and trained_with_obs_delay(config) and trained_with_ac_delay(config)),

        ("MAPPO, delay-oblivious",         lambda config: is_mappo(config) and trained_without_obs_delay(config) and trained_without_ac_delay(config)),
        ("MAPPO, delay-aware",             lambda config: is_mappo(config) and trained_with_obs_delay(config) and trained_with_ac_delay(config)),
]

line_colors = [
    '#56b4e9',  # light blue
    '#56b4e9',  # light blue
    '#4f67ab',  # dark blue
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
        ("\\texttt{Central-Multi}\\\\deployment", lambda config: config["runs"]["eval_depl"] == "cm"),
        ("\\texttt{Local-Multi}\\\\deployment", lambda config: config["runs"]["eval_depl"] == "lm"),
]

x_tick_name = "Delay scaling $\\lambda_{\\text{ac}}$"
x_tick_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
x_ticks = [(lambda config, v=v: config["eval_inference_delay_scaling"] == v) for v in x_tick_values]
x_tick_labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]

exp_names = ["0219_ac_delay"]
