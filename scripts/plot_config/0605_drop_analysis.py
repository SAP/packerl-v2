from ._base import *


sp_baseline_algo_cols = [
    ("\\spr", {
	    "m5_default": {
			"evaluation/metrics/Mean ttlDropped (MB)": 0.0, 
			"evaluation/metrics/Mean tcpDiscarded (MB)": 0.87197, 
			"evaluation/metrics/Mean macTxDropped (MB, gross)": 8.59523,
			},
		"b4_default": {
			"evaluation/metrics/Mean ttlDropped (MB)": 0.0, 
			"evaluation/metrics/Mean tcpDiscarded (MB)": 1.42563, 
			"evaluation/metrics/Mean macTxDropped (MB, gross)": 11.97463,
			},
		"nx_xs_default": {
			"evaluation/metrics/Mean ttlDropped (MB)": 0.0, 
			"evaluation/metrics/Mean tcpDiscarded (MB)": 2.11037, 
			"evaluation/metrics/Mean macTxDropped (MB, gross)": 23.03684,
			},
		"geant2001_default": {
			"evaluation/metrics/Mean ttlDropped (MB)": 0.0, 
			"evaluation/metrics/Mean tcpDiscarded (MB)": 3.33568, 
			"evaluation/metrics/Mean macTxDropped (MB, gross)": 49.90029,
			},
	}),  # RIP
    ("\\spe", {
	    "m5_default": {
			"evaluation/metrics/Mean ttlDropped (MB)": 0.0, 
			"evaluation/metrics/Mean tcpDiscarded (MB)": 0.97665, 
			"evaluation/metrics/Mean macTxDropped (MB, gross)": 8.75477, 
			},
		"b4_default": {
			"evaluation/metrics/Mean ttlDropped (MB)": 0.0, 
			"evaluation/metrics/Mean tcpDiscarded (MB)": 1.29408, 
			"evaluation/metrics/Mean macTxDropped (MB, gross)": 12.05806,
			},
		"nx_xs_default": {
			"evaluation/metrics/Mean ttlDropped (MB)": 0.0, 
			"evaluation/metrics/Mean tcpDiscarded (MB)": 2.14708, 
			"evaluation/metrics/Mean macTxDropped (MB, gross)": 23.22879,
			},
		"geant2001_default": {
			"evaluation/metrics/Mean ttlDropped (MB)": 0.0, 
			"evaluation/metrics/Mean tcpDiscarded (MB)": 3.18407, 
			"evaluation/metrics/Mean macTxDropped (MB, gross)": 50.09096,
			},
	}),  # EIGRP
]


def is_our_loggia(config):
    if config["runs"]["scenario"] in ["nx_xs", "geant2001"]:
        is_our_trainer = lambda config: config["runs"]["trainer"] == "il_ppo_c"
    else:
        is_our_trainer = lambda config: config["runs"]["trainer"] == "il_mappo_c" and config["runs"]["reward"] == "trace3_only_fast_decay"
    return config["exp_name"] == "0219_trainer" and is_our_trainer(config) and config["runs"]["eval_depl"] == "lm"
	
    
neural_algo_cols = [
    ("\\emph{MAGNNETO}", lambda config: config["exp_name"] == "0310_champs" and config["runs"]["algo"] == "magnneto"),
    ("\\emph{FieldLines}", lambda config: config["exp_name"] == "0310_champs" and config["runs"]["algo"] == "fl_tmlr24" and config["runs"]["trainer"] == "il_ppo_c"),
    ("\\emph{M-Slim}", lambda config:  config["exp_name"] == "0310_champs" and config["runs"]["algo"] == "ms_tmlr24" and config["runs"]["trainer"] == "il_ppo_c"),
    ("\\emph{LOGGIA} (ours)", lambda config: is_our_loggia(config)),
]


cols = sp_baseline_algo_cols + neural_algo_cols
col_colors = [
    '#949494',  # gray for RIP
	'#ca9161',  # brown for EIGRP
    '#fbafe4',  # pink
    '#d55e00',  # red orange
    '#de8f05',  # orange
    '#0173b2',  # blue
]
col_hatches = [None] * len(cols)
legend_labels = [col[0] for col in cols]
legend_algo_color_bucket_idx = range(len(cols))
legend_algo_hatches = col_hatches


plots = [
        ("\\emph{mini5}", lambda config: config["runs"]["full_scenario"] == "m5_default"),
        ("\\emph{B4}", lambda config: config["runs"]["full_scenario"] == "b4_default"),
        ("\\emph{nx-XS}", lambda config: config["runs"]["full_scenario"] == "nx_xs_default"),
        ("\\emph{GEANT}", lambda config: config["runs"]["full_scenario"] == "geant2001_default"),
]


drop_categories = [
    ("evaluation/metrics/Mean ttlDropped (MB)", "TTL Drop"),
    ("evaluation/metrics/Mean tcpDiscarded (MB)", "TCP Discard"),
    ("evaluation/metrics/Mean macTxDropped (MB, gross)", "Queue Drop"),
]


exp_names = ["0310_champs", "0219_trainer"]
