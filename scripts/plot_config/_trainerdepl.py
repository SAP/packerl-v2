from ._base import *

def is_none(config): return config["runs"]["trainer"] == "none"
def is_bc(config): return config["runs"]["trainer"] == "bc"
def is_il(config): return config["runs"]["trainer"] == "il"
def is_ppo_b(config): return config["runs"]["trainer"] == "ppo_b"
def is_ppo_c(config): return config["runs"]["trainer"] == "ppo_c"
def is_bc_ppo_b(config): return config["runs"]["trainer"] == "bc_ppo_b"
def is_bc_ppo_c(config): return config["runs"]["trainer"] == "bc_ppo_c"
def is_il_ppo_b(config): return config["runs"]["trainer"] == "il_ppo_b"
def is_il_ppo_c(config): return config["runs"]["trainer"] == "il_ppo_c"

cols = [("No Trainer", is_none),
        ("Behavior\nCloning", is_bc),
        ("DAgger-style\nIL", is_il),
        ("PPO\n(Birdseye observer)", is_ppo_b),
        ("PPO\n(Central observer)", is_ppo_c),
        ("IL → PPO\n(Central observer)", is_il_ppo_c),
]
col_name = "method"

def is_sb(config): return config["runs"]["eval_depl"] == "sb"
def is_sc(config):  return config["runs"]["eval_depl"] == "sc"
def is_mb(config): return config["runs"]["eval_depl"] == "mb"
def is_mc(config): return config["runs"]["eval_depl"] == "mc"
def is_ml(config): return config["runs"]["eval_depl"] == "ml"

rows = [("single-agent, birdseye obs.", is_sb),
        ("multi-agent, birdseye obs.", is_mb),
        ("single-agent, central obs.", is_sc),
        ("multi-agent, central obs.", is_mc),
        ("multi-agent, local obs.", is_ml),
]
row_name = "eval. depl."

exp_names = ["0130_m5_trainer"]