from ._base import *

marl_reward = [
        ("global_only", "global r"),
        ("mixed_trace3_fast_decay", "mixed r"),
        ("trace3_only_fast_decay", "L r")
]
marl_trainers = [
        ("MAPPO (C obs.)", "il_mappo_c"),
        ("MAPPO (L obs.)", "il_mappo_l"),
        ("IPPO (C obs.)", "il_ippo_c" ),
        ("IPPO (L obs.)", "il_ippo_l" ),
        ("HVPPO (C obs.)", "il_hvppo_c"),
        ("HVPPO (L obs.)", "il_hvppo_l"),
]
cols = [
        (f"{t[0]} [{r[1]}]", lambda config, t=t, r=r: config["runs"]["trainer"] == t[1] and config["runs"]["reward"] == r[0])
        for t in marl_trainers for r in marl_reward
]

col_colors = (["#4f67ab"] * 3 + ['#271f61'] * 3
              + ["#e3797e"] * 3 + ["#803337"] * 3
              + ["#67c266"] * 3 + ["#276926"] * 3 )
col_hatches = ["/", "\\", None] * 6

plot_baselines = [
        ("rip", 237.5695),
        ("rip", 329.4728),
        ("eigrp", 450.25678),
]

legend_algo_labels = ["MAPPO", "MAPPO (L)",
                      "IPPO", "IPPO (L)",
                      "HVPPO", "HVPPO (L)",
                      "Global reward", "Mixed reward", "Local reward"]
legend_algo_color_bucket_idx = [0, 3, 6, 9, 12, 15, -2, -2, -1]
legend_algo_hatches = [None, None, None, None, None, None,
                       "/", "\\", None]
legend_baseline_labels = list(set(b[0] for b in plot_baselines))

plot_limits = [
        (230, 243),  # mini5
        (320, 340),  # B4
        (447, 467),  # nx-XS
]

plots = [
        ("\\emph{mini5}", lambda config: config["runs"]["full_scenario"] == "m5_default"),
        ("\\emph{B4}", lambda config: config["runs"]["full_scenario"] == "b4_default"),
        ("\\emph{nx-XS}", lambda config: config["runs"]["full_scenario"] == "nx_xs_default"),
]
plot_alignment = "vertical"
run_filters = {"config.runs.eval_depl": "lm"}
exp_names = ["0219_trainer"]