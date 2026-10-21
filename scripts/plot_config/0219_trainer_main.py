from ._base import *

cols = [
        ("BC"            , lambda config: config["runs"]["trainer"] == "bc"),
        ("IL"            , lambda config: config["runs"]["trainer"] == "il"),
        ("PPO"           , lambda config: config["runs"]["trainer"] == "ppo_c"),
        ("BC → PPO"      , lambda config: config["runs"]["trainer"] == "bc_ppo_c"),
        ("IL → PPO"      , lambda config: config["runs"]["trainer"] == "il_ppo_c"),
        ("IL → MAPPO"    , lambda config: config["runs"]["trainer"] in ["il_mappo_c", "il_mappo_l"] and config["runs"]["reward"] == "trace3_only_fast_decay"),
]

col_colors = [
        '#949494',  # gray
        '#949494',  # gray
        '#029e73',  # green
        '#56b4e9',  # light blue
        '#56b4e9',  # light blue
        '#4f67ab',  # dark blue
]

col_hatches = [
        '//', None, None, '//', None, None
]

plot_baselines = [
        #("rip", 237.5695),
        ("rip", 329.4728),
        ("eigrp", 450.25678),
        ("eigrp", 441.8610),
]

legend_algo_labels = [c[0] for c in cols]
legend_algo_color_bucket_idx = [0, 0, 1, 2, 2, 3]
legend_algo_hatches = col_hatches
legend_baseline_labels = list(set(b[0] for b in plot_baselines))

plot_limits = [
        #(230, 243),  # mini5
        (300, 340),  # B4
        (427, 467),  # nx-XS
        (380, 480),  # GEANT
]

plots = [
        #("\\emph{mini5}", lambda config: config["runs"]["full_scenario"] == "m5_default"),
        ("\\emph{B4}", lambda config: config["runs"]["full_scenario"] == "b4_default"),
        ("\\emph{nx-XS}", lambda config: config["runs"]["full_scenario"] == "nx_xs_default"),
        ("\\emph{GEANT}", lambda config: config["runs"]["full_scenario"] == "geant2001_default"),
]
run_filters = {"config.runs.eval_depl": "lm"}
exp_names = ["0219_trainer"]