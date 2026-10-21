from ._base import *

all_lines = [
        ("\\texttt{Central-Single}\\\\deployment",  lambda config: config["runs"]["eval_depl"] == "cs"),
        ("\\texttt{Local-Multi}\\\\deployment",     lambda config: config["runs"]["eval_depl"] == "lm"),
]

line_colors = [
    '#bbbbbb',  # light gray
    '#555555',  # dark gray
]

line_markers = [
    'o', 'v'
]

line_dashes  = [
    None
] * len(all_lines)

all_plots = [
        ("Inference Times", lambda config: True),
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

scales_boxplot = 3, 100
baseline = "rip", 0
metric_name = "evaluation/times/Mean inference_time_ms"
exp_names = ["0316_scaling"]
