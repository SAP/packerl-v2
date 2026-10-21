from ._base import *

all_cols = [
        ("LOGGIA",         1, 0,      lambda config: config["runs"]["algo"] == "ms"),

        ("  —a",           1, 1,      lambda config: config["runs"]["algo"] == "ms_flat_space"),
        ("  —b",           1, 2,      lambda config: config["runs"]["algo"] == "ms_line_graph"),
        ("  —c",           1, 3,      lambda config: config["runs"]["algo"] == "ms_small_nn"),

        ("  —d",           1, 4,      lambda config: config["runs"]["algo"] == "ms_full_pi_updates"),
        ("  —e",           1, 5,      lambda config: config["runs"]["algo"] == "ms_old_exploration"),
        ("  +f",           1, 6,      lambda config: config["runs"]["algo"] == "ms_vf_onehot_infuse"),
        ("  +g",           1, 7,      lambda config: config["runs"]["algo"] == "ms_vf_meta"),

        ("M-Slim",         3, 0,      lambda config: config["runs"]["algo"] == "ms_tmlr24"),
        ("  +a",           3, 1,      lambda config: config["runs"]["algo"] == "ms_tmlr24_log_space"),
        ("  +b",           3, 2,      lambda config: config["runs"]["algo"] == "ms_tmlr24_base_graph"),
        ("  +c",           3, 3,      lambda config: config["runs"]["algo"] == "ms_tmlr24_big_nn"),

        ("  +d",           3, 4,      lambda config: config["runs"]["algo"] == "ms_tmlr24_adaptive_pi_updates"),
        ("  +e",           3, 5,      lambda config: config["runs"]["algo"] == "ms_tmlr24_new_exploration"),

        ("  +p",           5, 0,      lambda config: config["runs"]["algo"] == "ms_path"),
        ("  +p +r3",       5, 1,      lambda config: config["runs"]["algo"] == "ms_path_rand3"),
        ("  +p +r3 +trpl", 5, 2,      lambda config: config["runs"]["algo"] == "ms_path_rand3_trpl"),
]
col_name = "Algorithm"

exp_names = ["0217_ms_ablation"]