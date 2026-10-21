from ._base import *

def is_our_loggia(config):
    if config["runs"]["scenario"] in ["nx_xs", "geant2001"]:
        is_our_trainer = lambda config: config["runs"]["trainer"] == "il_ppo_c"
    else:
        is_our_trainer = lambda config: config["runs"]["trainer"] == "il_mappo_c" and config["runs"]["reward"] == "trace3_only_fast_decay"
    return config["exp_name"] == "0219_trainer" and is_our_trainer(config) and config["runs"]["eval_depl"] == "lm"

cols = [
    ("\\emph{MAGNNETO}", lambda config: config["exp_name"] == "0310_champs" and config["runs"]["algo"] == "magnneto"),
    ("\\emph{FieldLines}", lambda config: config["exp_name"] == "0310_champs" and config["runs"]["algo"] == "fl_tmlr24" and config["runs"]["trainer"] == "il_ppo_c"),
    ("\\emph{M-Slim}", lambda config:  config["exp_name"] == "0310_champs" and config["runs"]["algo"] == "ms_tmlr24" and config["runs"]["trainer"] == "il_ppo_c"),
    ("\\emph{LOGGIA} (ours)", lambda config: is_our_loggia(config)),
]

col_colors = [
    '#fbafe4',  # pink
    '#d55e00',  # red orange
    '#de8f05',  # orange
    '#0173b2',  # blue
]
col_hatches = [None] * len(col_colors)

plot_baselines = [
    ("rip", 237.5695),
    ("rip", 329.4728),
    ("eigrp", 442.7223),
    ("eigrp", 441.8610),
]

legend_algo_labels = [c[0] for c in cols]
legend_algo_color_bucket_idx = range(len(cols))
legend_algo_hatches = col_hatches
legend_baseline_labels = list(set(b[0] for b in plot_baselines))

plot_limits = [
        (0, 225, 245),  # mini5
        (120, 300, 340),  # B4
        (0, 350, 470),  # nx-XS
        (280, 400, 490),  # GEANT
]

plots = [
        ("\\emph{mini5}", lambda config: config["runs"]["full_scenario"] == "m5_default"),
        ("\\emph{B4}", lambda config: config["runs"]["full_scenario"] == "b4_default"),
        ("\\emph{nx-XS}", lambda config: config["runs"]["full_scenario"] == "nx_xs_default"),
        ("\\emph{GEANT}", lambda config: config["runs"]["full_scenario"] == "geant2001_default"),
]

exp_names = ["0310_champs", "0219_trainer"]
