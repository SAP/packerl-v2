from ._base import *


def is_our_loggia(config):
    if config["runs"]["scenario"] in ["nx_xs", "geant2001"]:
        is_our_trainer = lambda config: config["runs"]["trainer"] == "il_ppo_c"
    else:
        is_our_trainer = lambda config: config["runs"]["trainer"] == "il_mappo_c" and config["runs"]["reward"] == "trace3_only_fast_decay"
    return config["exp_name"] == "0219_trainer" and is_our_trainer(config) and config["runs"]["eval_depl"] == "lm"

all_cols = [
    ("MAGNNETO", lambda config: config["exp_name"] == "0310_champs" and config["runs"]["algo"] == "magnneto"),
    ("FieldLines", lambda config: config["exp_name"] == "0310_champs" and config["runs"]["algo"] == "fl_tmlr24" and config["runs"]["trainer"] == "il_ppo_c"),
    ("M-Slim", lambda config:  config["exp_name"] == "0310_champs" and config["runs"]["algo"] == "ms_tmlr24" and config["runs"]["trainer"] == "il_ppo_c"),
    ("LOGGIA (ours)", lambda config: is_our_loggia(config)),
]

primary_metric = metric_name
baselines = {
    "mini5": ("rip", {primary_metric: 237.5695,
                      "evaluation/metrics/Mean avgDelay (ms)": 9.5007,
                      "evaluation/metrics/Mean dropRatio (%)": 0.3671,
                      "evaluation/metrics/Mean avgJitter (ms)": 0.1195,
                      "evaluation/metrics/Mean avgTxQueueRelativeLoad (%)": 24.3822,
                      "evaluation/metrics/Mean tcpDiscarded (MB)": 0.8720,
                      }),
    "B4": ("rip", {primary_metric: 329.4728,
                    "evaluation/metrics/Mean avgDelay (ms)": 11.3454,
                    "evaluation/metrics/Mean dropRatio (%)": 0.4296,
                    "evaluation/metrics/Mean avgJitter (ms)": 0.1357,
                    "evaluation/metrics/Mean avgTxQueueRelativeLoad (%)": 13.9608,
                    "evaluation/metrics/Mean tcpDiscarded (MB)": 1.4256,
                   }),
    "nx-XS": ("rip", {primary_metric: 442.7223,
                      "evaluation/metrics/Mean avgDelay (ms)": 9.4499,
                      "evaluation/metrics/Mean dropRatio (%)": 0.4856,
                      "evaluation/metrics/Mean avgJitter (ms)": 0.1211,
                      "evaluation/metrics/Mean avgTxQueueRelativeLoad (%)": 23.4904,
                      "evaluation/metrics/Mean tcpDiscarded (MB)": 2.1104,
                      }),
    "GEANT": ("eigrp", {primary_metric: 441.8611,
                        "evaluation/metrics/Mean avgDelay (ms)": 9.3750,
                        "evaluation/metrics/Mean dropRatio (%)": 0.7153,
                        "evaluation/metrics/Mean avgJitter (ms)": 0.1834,
                        "evaluation/metrics/Mean avgTxQueueRelativeLoad (%)": 16.3615,
                        "evaluation/metrics/Mean tcpDiscarded (MB)": 3.1841,
                        }),
}

all_rows = [
    ("Delivered (MB) ↑", primary_metric),
    ("Latency (ms) ↓", "evaluation/metrics/Mean avgDelay (ms)"),
    ("Queue Load (%) ↓", "evaluation/metrics/Mean avgTxQueueRelativeLoad (%)"),
    ("TCP Discards (MB) ↓", "evaluation/metrics/Mean tcpDiscarded (MB)"),
]

all_plots = [
    ("mini5", lambda config: config["runs"]["full_scenario"] == "m5_default"),
    ("B4", lambda config: config["runs"]["full_scenario"] == "b4_default"),
    ("nx-XS", lambda config: config["runs"]["full_scenario"] == "nx_xs_default"),
    ("GEANT", lambda config: config["runs"]["full_scenario"] == "geant2001_default"),
]

exp_names = ["0310_champs", "0219_trainer"]