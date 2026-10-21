"""
Test for the OnPolicyBuffer class in the RL PPO algorithm, mainly focusing on return and advantage calculations.
"""
import sys
sys.path.append(".")

import pytest

import torch

from src.trainer.ppo.utils.ret_adv import compute_returns_and_advantages
from src.utils.constants import BIRDSEYE_OBSERVER_ID


# Helper function to bootstrap the terminal reward. Taken from PPO implementation and adapted to run outside the class.
def bootstrap_terminal_reward(reward, terminal_value, discount_factor, value_loc):
    """
    Bootstraps the terminal reward of an episode with the final value of the last observation.
    """
    reward_values, reward_observer_idx = reward
    is_global = reward_observer_idx == BIRDSEYE_OBSERVER_ID
    if value_loc == "single":
        # policy returned global terminal value; local rewards won't be used and can be left unchanged
        reward_values[is_global] = reward_values[is_global] + discount_factor * terminal_value
    else:
        # policy returned local terminal values; global rewards won't be used and can be left unchanged
        reward_values[~is_global] = reward_values[~is_global] + discount_factor * terminal_value
    return reward_values, reward_observer_idx

# === deployment /buffer args ================================================================================

func_args_ss = {"actor_loc": "single", "value_loc": "single"}
func_args_ms = {"actor_loc": "multi", "value_loc": "single"}
func_args_mm = {"actor_loc": "multi", "value_loc": "multi"}

comp_args_g0_d0 = {"gae_lambda": 0.0, "discount_factor": 0.0}
comp_args_g0_d1 = {"gae_lambda": 0.0, "discount_factor": 1.0}
comp_args_g1_d1 = {"gae_lambda": 1.0, "discount_factor": 1.0}
comp_args_gh_dh = {"gae_lambda": 0.8, "discount_factor": 0.5}

# === experience sequences =======================================================

# each experience sequence consists of a list of step information dicts ("observers", "values", "rewards", "done"),
# and a final dict with terminal information ("terminal_observers", "terminal_values").
# observer IDs are sometimes out of order to check whether the function can handle it.
# note that rewards are provided for the observers of the NEXT step, not the current one.

# 1) Single-agent (agent 2), 1‐step episode
seq1_single_singlestep = (
    [
        {"observers": [-1, 2], "values": [0.0, 1.0],
         "next_observers": [2, -1], "rewards": [0.0, 1.0],
         "done": True},
    ],
    {
        "last_values": [1.0, 1.0]
    }
)

# 2) Single-agent (agent 2), 2‐step episode, zero‐reward baseline (all r=0, V≠0)
seq2_single_zero_reward = (
    [
        {"observers": [2, -1], "values": [1.0, -1.0],
         "next_observers": [2, -1], "rewards": [0.0, 0.0],
         "done": False},
        {"observers": [2, -1], "values": [2.0, -1.0],
         "next_observers": [2, -1], "rewards": [0.0, 0.0],
         "done": True},
    ],
    {
        "last_values": [0.0, 1.0],
    }
)

# 3) Single-agent (agent 2), 2‐step episode, zero‐value (all V=0, r≠0)
seq3_single_zero_value = (
    [
        {"observers": [2, -1],  "values": [0.0, 0.0],
         "next_observers": [2, -1], "rewards": [1.0, 0.0],
         "done": False},
        {"observers": [2, -1], "values": [0.0, 0.0],
         "next_observers": [-1, 2], "rewards": [4.0, 2.0],
         "done": True,},
    ],
    {
        "last_values": [0.0, 0.0],
    }
)

# 5) Multi-agent (agents 2 and 5), 1‐step episode
seq5_multi_singlestep = (
    [
        {"observers": [-1, 5, 2], "values": [1.0, -1.0, 0.0],
         "next_observers": [2, 5, -1], "rewards": [1.0, 2.0, 0.0],
         "done": True,},
    ],
    {
        "last_values": [8.0, 1.0, 8.0],
    }
)

# 6) Multi‐agent (agents 2 and 5), 2‐step episode
seq6_multi_multistep = (
    [
        {"observers": [-1, 2, 5], "values": [1.0, 2.0, 2.0],
         "next_observers": [-1, 5, 2], "rewards": [1.0, 2.0, 0.0],
         "done": False,},
        {"observers": [-1, 5, 2], "values": [2.0, 4.0, 0.0],
         "next_observers": [2, 5, -1], "rewards": [8.0, 4.0, 4.0],
         "done": True,},
    ],
    {
        "last_values": [2.0, 0.0, 0.0],
    }
)

# 7) Multi‐agent (agents 2 and 5), 2‐step episode, agent 5 disappears in 1st step
seq7_disappearance_1st = (
    [
        {"observers": [-1, 2, 5], "values": [1.0, 2.0, 2.0],
         "next_observers": [-1, 2], "rewards": [1.0, 0.0],
         "done": False,},
        {"observers": [-1, 2], "values": [2.0, 0.0],
         "next_observers": [2, -1], "rewards": [8.0, 4.0],
         "done": True,},
    ],
    {
        "last_values": [2.0, 0.0],
    }
)

# 8) Multi-agent (agents 2 and 5), 2‐step episode, agent 5 disappears in 2nd step
seq8_disappearance_2nd = (
    [
        {"observers": [-1, 2, 5], "values": [1.0, 2.0, 2.0],
         "next_observers": [-1, 5, 2], "rewards": [1.0, 2.0, 0.0],
         "done": False,},
        {"observers": [-1, 5, 2], "values": [2.0, 4.0, 0.0],
         "next_observers": [2, -1], "rewards": [8.0, 4.0],
         "done": True,},
    ],
    {
        "last_values": [2.0, 0.0],
    }
)

# 9) Multi-agent (agents 2 and 5), 2‐step episode, agent 5 disappears in 1st step and reappears in last step
seq9_gone_and_return = (
    [
        {"observers": [-1, 2, 5], "values": [1.0, 2.0, 2.0],
         "next_observers": [-1, 2], "rewards": [1.0, 0.0],
         "done": False,},
        {"observers": [-1, 2], "values": [2.0, 0.0],
         "next_observers": [2, 5, -1], "rewards": [8.0, 4.0, 4.0],
         "done": True,},
    ],
    {
        "last_values": [2.0, 3.0, 0.0],
    }
)

# 10) Multi-agent Two 2‐step episodes:
#     ep 1: starts with agents 2 & 5, then 5 disappears in 2nd step (like seq8_disappearance_2nd)
#     ep 2: starts with agent 2 & 5, then new agent 13 appears in 1st step
seq10_two_episodes = (
    [
        # episode 1
        {"observers": [-1, 2, 5], "values": [1.0, 2.0, 2.0],
         "next_observers": [-1, 5, 2], "rewards": [1.0, 2.0, 0.0],
         "done": False,},
        {"observers": [-1, 5, 2], "values": [2.0, 4.0, 0.0],
         "next_observers": [2, -1], "rewards": [8.0, 4.0],
         "done": True,},
        # episode 2

        {"observers": [-1, 2, 5], "values": [1.0, 2.0, 2.0],
         "next_observers": [-1, 5, 2, 13], "rewards": [1.0, 2.0, 0.0, 3.0],
         "done": False, },
        {"observers": [-1, 5, 13, 2], "values": [2.0, 4.0, -1.0, 0.0],
         "next_observers": [2, 5, -1, 13], "rewards": [8.0, 4.0, 4.0, 4.0],
         "done": True, },
    ],
    {
        "last_values": [2.0, 0.0, 0.0, 1.0],
    }
)

# ===== Expected returns and advantages: Sequence 1 ===================================================================

# --- Single-agent, single-value -------------------------------------------------------------

expected_seq1_ss_g0_d0 = {
    "advantages": [
        [1]
    ],
    "returns": [
        [1]
    ],
}

expected_seq1_ss_g0_d1 = {
    "advantages": [
        [1]
    ],
    "returns": [
        [1]
    ],
}

expected_seq1_ss_g1_d1 = {
    "advantages": [
        [1]
    ],
    "returns": [
        [1]
    ],
}

expected_seq1_ss_gh_dh = {
    "advantages": [
        [1]
    ],
    "returns": [
        [1]
    ],
}

# --- Multi-agent, single-value -------------------------------------------------------------

expected_seq1_ms_g0_d0 = {
    "advantages": [
        [0]
    ],
    "returns": [
        [0]
    ],
}

expected_seq1_ms_g0_d1 = {
    "advantages": [
        [0]
    ],
    "returns": [
        [0]
    ],
}

expected_seq1_ms_g1_d1 = {
    "advantages": [
        [0]
    ],
    "returns": [
        [0]
    ],
}

expected_seq1_ms_gh_dh = {
    "advantages": [
        [0]
    ],
    "returns": [
        [0]
    ],
}

# --- Multi-agent, multi-value -------------------------------------------------------------

expected_seq1_mm_g0_d0 = {
    "advantages": [
        [-1]
    ],
    "returns": [
        [0]
    ],
}

expected_seq1_mm_g0_d1 = {
    "advantages": [
        [-1]
    ],
    "returns": [
        [0]
    ],
}

expected_seq1_mm_g1_d1 = {
    "advantages": [
        [-1]
    ],
    "returns": [
        [0]
    ],
}

expected_seq1_mm_gh_dh = {
    "advantages": [
        [-1]
    ],
    "returns": [
        [0]
    ],
}

# --- Settings for Sequence 1 -------------------------------------------------------------

settings_seq1 = [
    # single-agent, single-value
    (seq1_single_singlestep, func_args_ss, comp_args_g0_d0, expected_seq1_ss_g0_d0),
    (seq1_single_singlestep, func_args_ss, comp_args_g0_d1, expected_seq1_ss_g0_d1),
    (seq1_single_singlestep, func_args_ss, comp_args_g1_d1, expected_seq1_ss_g1_d1),
    (seq1_single_singlestep, func_args_ss, comp_args_gh_dh, expected_seq1_ss_gh_dh),
    # multi-agent, single-value
    (seq1_single_singlestep, func_args_ms, comp_args_g0_d0, expected_seq1_ms_g0_d0),
    (seq1_single_singlestep, func_args_ms, comp_args_g0_d1, expected_seq1_ms_g0_d1),
    (seq1_single_singlestep, func_args_ms, comp_args_g1_d1, expected_seq1_ms_g1_d1),
    (seq1_single_singlestep, func_args_ms, comp_args_gh_dh, expected_seq1_ms_gh_dh),
    # multi-agent, multi-value
    (seq1_single_singlestep, func_args_mm, comp_args_g0_d0, expected_seq1_mm_g0_d0),
    (seq1_single_singlestep, func_args_mm, comp_args_g0_d1, expected_seq1_mm_g0_d1),
    (seq1_single_singlestep, func_args_mm, comp_args_g1_d1, expected_seq1_mm_g1_d1),
    (seq1_single_singlestep, func_args_mm, comp_args_gh_dh, expected_seq1_mm_gh_dh),
]

# ===== Expected returns and advantages: Sequence 2 ===================================================================

# --- Single-agent, single-value -------------------------------------------------------------

expected_seq2_ss_g0_d0 = {
    "advantages": [
        [1],
        [1]
    ],
    "returns": [
        [0],
        [0]
    ],
}

expected_seq2_ss_g0_d1 = {
    "advantages": [
        [0],
        [1]
    ],
    "returns": [
        [-1],
        [0]
    ],
}

expected_seq2_ss_g1_d1 = {
    "advantages": [
        [1],
        [1]
    ],
    "returns": [
        [0],
        [0]
    ],
}

expected_seq2_ss_gh_dh = {
    "advantages": [
        [0.9],
        [1]
    ],
    "returns": [
        [-0.1],
        [0]
    ],
}

# --- Multi-agent, single-value -------------------------------------------------------------

expected_seq2_ms_g0_d0 = {
    "advantages": [
        [1],
        [1]
    ],
    "returns": [
        [0],
        [0]
    ],
}

expected_seq2_ms_g0_d1 = {
    "advantages": [
        [0],
        [1]
    ],
    "returns": [
        [-1],
        [0]
    ],
}

expected_seq2_ms_g1_d1 = {
    "advantages": [
        [1],
        [1]
    ],
    "returns": [
        [0],
        [0]
    ],
}

expected_seq2_ms_gh_dh = {
    "advantages": [
        [0.9],
        [1]
    ],
    "returns": [
        [-0.1],
        [0]
    ],
}

# --- Multi-agent, multi-value -------------------------------------------------------------

expected_seq2_mm_g0_d0 = {
    "advantages": [
        [-1],
        [-2]
    ],
    "returns": [
        [0],
        [0]
    ],
}

expected_seq2_mm_g0_d1 = {
    "advantages": [
        [1],
        [-2]
    ],
    "returns": [
        [2],
        [0]
    ],
}

expected_seq2_mm_g1_d1 = {
    "advantages": [
        [-1],
        [-2]
    ],
    "returns": [
        [0],
        [0]
    ],
}

expected_seq2_mm_gh_dh = {
    "advantages": [
        [-0.8],
        [-2]
    ],
    "returns": [
        [0.2],
        [0]
    ],
}

# --- Settings for Sequence 2 -------------------------------------------------------------

settings_seq2 = [
    # single-agent, single-value
    (seq2_single_zero_reward, func_args_ss, comp_args_g0_d0, expected_seq2_ss_g0_d0),
    (seq2_single_zero_reward, func_args_ss, comp_args_g0_d1, expected_seq2_ss_g0_d1),
    (seq2_single_zero_reward, func_args_ss, comp_args_g1_d1, expected_seq2_ss_g1_d1),
    (seq2_single_zero_reward, func_args_ss, comp_args_gh_dh, expected_seq2_ss_gh_dh),
    # multi-agent, single-value
    (seq2_single_zero_reward, func_args_ms, comp_args_g0_d0, expected_seq2_ms_g0_d0),
    (seq2_single_zero_reward, func_args_ms, comp_args_g0_d1, expected_seq2_ms_g0_d1),
    (seq2_single_zero_reward, func_args_ms, comp_args_g1_d1, expected_seq2_ms_g1_d1),
    (seq2_single_zero_reward, func_args_ms, comp_args_gh_dh, expected_seq2_ms_gh_dh),
    # multi-agent, multi-value
    (seq2_single_zero_reward, func_args_mm, comp_args_g0_d0, expected_seq2_mm_g0_d0),
    (seq2_single_zero_reward, func_args_mm, comp_args_g0_d1, expected_seq2_mm_g0_d1),
    (seq2_single_zero_reward, func_args_mm, comp_args_g1_d1, expected_seq2_mm_g1_d1),
    (seq2_single_zero_reward, func_args_mm, comp_args_gh_dh, expected_seq2_mm_gh_dh),
]

# ===== Expected returns and advantages: Sequence 3 ===================================================================

# --- Single-agent, single-value -------------------------------------------------------------

expected_seq3_ss_g0_d0 = {
    "advantages": [
        [0],
        [4]
    ],
    "returns": [
        [0],
        [4]
    ],
}

expected_seq3_ss_g0_d1 = {
    "advantages": [
        [0],
        [4]
    ],
    "returns": [
        [0],
        [4]
    ],
}

expected_seq3_ss_g1_d1 = {
    "advantages": [
        [4],
        [4]
    ],
    "returns": [
        [4],
        [4]
    ],
}

expected_seq3_ss_gh_dh = {
    "advantages": [
        [1.6],
        [4]
    ],
    "returns": [
        [1.6],
        [4]
    ],
}

# --- Multi-agent, single-value -------------------------------------------------------------

expected_seq3_ms_g0_d0 = {
    "advantages": [
        [1],
        [2]
    ],
    "returns": [
        [1],
        [2]
    ],
}

expected_seq3_ms_g0_d1 = {
    "advantages": [
        [1],
        [2]
    ],
    "returns": [
        [1],
        [2]
    ],
}

expected_seq3_ms_g1_d1 = {
    "advantages": [
        [3],
        [2]
    ],
    "returns": [
        [3],
        [2]
    ],
}

expected_seq3_ms_gh_dh = {
    "advantages": [
        [1.8],
        [2]
    ],
    "returns": [
        [1.8],
        [2]
    ],
}

# --- Multi-agent, multi-value -------------------------------------------------------------

expected_seq3_mm_g0_d0 = {
    "advantages": [
        [1],
        [2]
    ],
    "returns": [
        [1],
        [2]
    ],
}

expected_seq3_mm_g0_d1 = {
    "advantages": [
        [1],
        [2]
    ],
    "returns": [
        [1],
        [2]
    ],
}

expected_seq3_mm_g1_d1 = {
    "advantages": [
        [3],
        [2]
    ],
    "returns": [
        [3],
        [2]
    ],
}

expected_seq3_mm_gh_dh = {
    "advantages": [
        [1.8],
        [2]
    ],
    "returns": [
        [1.8],
        [2]
    ],
}

# --- Settings for Sequence 3 -------------------------------------------------------------

settings_seq3 = [
    # single-agent, single-value
    (seq3_single_zero_value, func_args_ss, comp_args_g0_d0, expected_seq3_ss_g0_d0),
    (seq3_single_zero_value, func_args_ss, comp_args_g0_d1, expected_seq3_ss_g0_d1),
    (seq3_single_zero_value, func_args_ss, comp_args_g1_d1, expected_seq3_ss_g1_d1),
    (seq3_single_zero_value, func_args_ss, comp_args_gh_dh, expected_seq3_ss_gh_dh),
    # multi-agent, single-value
    (seq3_single_zero_value, func_args_ms, comp_args_g0_d0, expected_seq3_ms_g0_d0),
    (seq3_single_zero_value, func_args_ms, comp_args_g0_d1, expected_seq3_ms_g0_d1),
    (seq3_single_zero_value, func_args_ms, comp_args_g1_d1, expected_seq3_ms_g1_d1),
    (seq3_single_zero_value, func_args_ms, comp_args_gh_dh, expected_seq3_ms_gh_dh),
    # multi-agent, multi-value
    (seq3_single_zero_value, func_args_mm, comp_args_g0_d0, expected_seq3_mm_g0_d0),
    (seq3_single_zero_value, func_args_mm, comp_args_g0_d1, expected_seq3_mm_g0_d1),
    (seq3_single_zero_value, func_args_mm, comp_args_g1_d1, expected_seq3_mm_g1_d1),
    (seq3_single_zero_value, func_args_mm, comp_args_gh_dh, expected_seq3_mm_gh_dh),
]

# ===== Expected returns and advantages: Sequence 5 ===================================================================

# --- Single-agent, single-value -------------------------------------------------------------

expected_seq5_ss_g0_d0 = {
    "advantages": [
        [-1]
    ],
    "returns": [
        [0]
    ],
}

expected_seq5_ss_g0_d1 = {
    "advantages": [
        [-1]
    ],
    "returns": [
        [0]
    ],
}

expected_seq5_ss_g1_d1 = {
    "advantages": [
        [-1]
    ],
    "returns": [
        [0]
    ],
}

expected_seq5_ss_gh_dh = {
    "advantages": [
        [-1]
    ],
    "returns": [
        [0]
    ],
}

# --- Multi-agent, single-value -------------------------------------------------------------

expected_seq5_ms_g0_d0 = {
    "advantages": [
        [0, 1]
    ],
    "returns": [
        [1, 2]
    ],
}

expected_seq5_ms_g0_d1 = {
    "advantages": [
        [0, 1]
    ],
    "returns": [
        [1, 2]
    ],
}

expected_seq5_ms_g1_d1 = {
    "advantages": [
        [0, 1]
    ],
    "returns": [
        [1, 2]
    ],
}

expected_seq5_ms_gh_dh = {
    "advantages": [
        [0, 1]
    ],
    "returns": [
        [1, 2]
    ],
}

# --- Multi-agent, multi-value -------------------------------------------------------------

expected_seq5_mm_g0_d0 = {
    "advantages": [
        [1, 3]
    ],
    "returns": [
        [1, 2]
    ],
}

expected_seq5_mm_g0_d1 = {
    "advantages": [
        [1, 3]
    ],
    "returns": [
        [1, 2]
    ],
}
expected_seq5_mm_g1_d1 = {
    "advantages": [
        [1, 3]
    ],
    "returns": [
        [1, 2]
    ],
}

expected_seq5_mm_gh_dh = {
    "advantages": [
        [1, 3]
    ],
    "returns": [
        [1, 2]
    ],
}

# --- Settings for Sequence 5 -------------------------------------------------------------

settings_seq5 = [
    # single-agent, single-value
    (seq5_multi_singlestep, func_args_ss, comp_args_g0_d0, expected_seq5_ss_g0_d0),
    (seq5_multi_singlestep, func_args_ss, comp_args_g0_d1, expected_seq5_ss_g0_d1),
    (seq5_multi_singlestep, func_args_ss, comp_args_g1_d1, expected_seq5_ss_g1_d1),
    (seq5_multi_singlestep, func_args_ss, comp_args_gh_dh, expected_seq5_ss_gh_dh),
    # multi-agent, single-value
    (seq5_multi_singlestep, func_args_ms, comp_args_g0_d0, expected_seq5_ms_g0_d0),
    (seq5_multi_singlestep, func_args_ms, comp_args_g0_d1, expected_seq5_ms_g0_d1),
    (seq5_multi_singlestep, func_args_ms, comp_args_g1_d1, expected_seq5_ms_g1_d1),
    (seq5_multi_singlestep, func_args_ms, comp_args_gh_dh, expected_seq5_ms_gh_dh),
    # multi-agent, multi-value
    (seq5_multi_singlestep, func_args_mm, comp_args_g0_d0, expected_seq5_mm_g0_d0),
    (seq5_multi_singlestep, func_args_mm, comp_args_g0_d1, expected_seq5_mm_g0_d1),
    (seq5_multi_singlestep, func_args_mm, comp_args_g1_d1, expected_seq5_mm_g1_d1),
    (seq5_multi_singlestep, func_args_mm, comp_args_gh_dh, expected_seq5_mm_gh_dh),
]

# ===== Expected returns and advantages: Sequence 6 ===================================================================

# --- Single-agent, single-value -------------------------------------------------------------

expected_seq6_ss_g0_d0 = {
    "advantages": [
        [0],
        [2]
    ],
    "returns": [
        [1],
        [4]
    ],
}

expected_seq6_ss_g0_d1 = {
    "advantages": [
        [2],
        [2]
    ],
    "returns": [
        [3],
        [4]
    ],
}

expected_seq6_ss_g1_d1 = {
    "advantages": [
        [4],
        [2]
    ],
    "returns": [
        [5],
        [4]
    ],
}

expected_seq6_ss_gh_dh = {
    "advantages": [
        [1.8],
        [2]
    ],
    "returns": [
        [2.8],
        [4]
    ],
}

# --- Multi-agent, single-value -------------------------------------------------------------

expected_seq6_ms_g0_d0 = {
    "advantages": [
        [-1, 1],
        [6, 2]
    ],
    "returns": [
        [0, 2],
        [8, 4]
    ],
}

expected_seq6_ms_g0_d1 = {
    "advantages": [
        [1, 3],
        [6, 2]
    ],
    "returns": [
        [2, 4],
        [8, 4]
    ],
}

expected_seq6_ms_g1_d1 = {
    "advantages": [
        [7, 5],
        [6, 2]
    ],
    "returns": [
        [8, 6],
        [8, 4]
    ],
}

expected_seq6_ms_gh_dh = {
    "advantages": [
        [2.4, 2.8],
        [6, 2]
    ],
    "returns": [
        [3.4, 3.8],
        [8, 4]
    ],
}

# --- Multi-agent, multi-value -------------------------------------------------------------

expected_seq6_mm_g0_d0 = {
    "advantages": [
        [-2, 0],
        [8, 0]
    ],
    "returns": [
        [0, 2],
        [8, 4]
    ],
}

expected_seq6_mm_g0_d1 = {
    "advantages": [
        [-2, 4],
        [8, 0]
    ],
    "returns": [
        [0, 6],
        [8, 4]
    ],
}

expected_seq6_mm_g1_d1 = {
    "advantages": [
        [6, 4],
        [8, 0]
    ],
    "returns": [
        [8, 6],
        [8, 4]
    ],
}

expected_seq6_mm_gh_dh = {
    "advantages": [
        [1.2, 2],
        [8, 0]
    ],
    "returns": [
        [3.2, 4],
        [8, 4]
    ],
}

# --- Settings for Sequence 6 -------------------------------------------------------------

settings_seq6 = [
    # single-agent, single-value
    (seq6_multi_multistep, func_args_ss, comp_args_g0_d0, expected_seq6_ss_g0_d0),
    (seq6_multi_multistep, func_args_ss, comp_args_g0_d1, expected_seq6_ss_g0_d1),
    (seq6_multi_multistep, func_args_ss, comp_args_g1_d1, expected_seq6_ss_g1_d1),
    (seq6_multi_multistep, func_args_ss, comp_args_gh_dh, expected_seq6_ss_gh_dh),
    # multi-agent, single-value
    (seq6_multi_multistep, func_args_ms, comp_args_g0_d0, expected_seq6_ms_g0_d0),
    (seq6_multi_multistep, func_args_ms, comp_args_g0_d1, expected_seq6_ms_g0_d1),
    (seq6_multi_multistep, func_args_ms, comp_args_g1_d1, expected_seq6_ms_g1_d1),
    (seq6_multi_multistep, func_args_ms, comp_args_gh_dh, expected_seq6_ms_gh_dh),
    # multi-agent, multi-value
    (seq6_multi_multistep, func_args_mm, comp_args_g0_d0, expected_seq6_mm_g0_d0),
    (seq6_multi_multistep, func_args_mm, comp_args_g0_d1, expected_seq6_mm_g0_d1),
    (seq6_multi_multistep, func_args_mm, comp_args_g1_d1, expected_seq6_mm_g1_d1),
    (seq6_multi_multistep, func_args_mm, comp_args_gh_dh, expected_seq6_mm_gh_dh),
]

# ===== Expected returns and advantages: Sequence 7 ===================================================================

# --- Multi-agent, single-value -------------------------------------------------------------

expected_seq7_ms_g0_d0 = {
    "advantages": [
        [-1],
        [6]
    ],
    "returns": [
        [0],
        [8]
    ],
}

expected_seq7_ms_g0_d1 = {
    "advantages": [
        [1],
        [6]
    ],
    "returns": [
        [2],
        [8]
    ],
}

expected_seq7_ms_g1_d1 = {
    "advantages": [
        [7],
        [6]
    ],
    "returns": [
        [8],
        [8]
    ],
}

expected_seq7_ms_gh_dh = {
    "advantages": [
        [2.4],
        [6]
    ],
    "returns": [
        [3.4],
        [8]
    ],
}

# --- Multi-agent, multi-value -------------------------------------------------------------

expected_seq7_mm_g0_d0 = {
    "advantages": [
        [-2],
        [8]
    ],
    "returns": [
        [0],
        [8]
    ],
}

expected_seq7_mm_g0_d1 = {
    "advantages": [
        [-2],
        [8]
    ],
    "returns": [
        [0],
        [8]
    ],
}

expected_seq7_mm_g1_d1 = {
    "advantages": [
        [6],
        [8]
    ],
    "returns": [
        [8],
        [8]
    ],
}

expected_seq7_mm_gh_dh = {
    "advantages": [
        [1.2],
        [8]
    ],
    "returns": [
        [3.2],
        [8]
    ],
}

# --- Settings for Sequence 7 -------------------------------------------------------------

settings_seq7 = [
    # single-agent, single-value is the same as in sequence 6
    # multi-agent, single-value
    (seq7_disappearance_1st, func_args_ms, comp_args_g0_d0, expected_seq7_ms_g0_d0),
    (seq7_disappearance_1st, func_args_ms, comp_args_g0_d1, expected_seq7_ms_g0_d1),
    (seq7_disappearance_1st, func_args_ms, comp_args_g1_d1, expected_seq7_ms_g1_d1),
    (seq7_disappearance_1st, func_args_ms, comp_args_gh_dh, expected_seq7_ms_gh_dh),
    # multi-agent, multi-value
    (seq7_disappearance_1st, func_args_mm, comp_args_g0_d0, expected_seq7_mm_g0_d0),
    (seq7_disappearance_1st, func_args_mm, comp_args_g0_d1, expected_seq7_mm_g0_d1),
    (seq7_disappearance_1st, func_args_mm, comp_args_g1_d1, expected_seq7_mm_g1_d1),
    (seq7_disappearance_1st, func_args_mm, comp_args_gh_dh, expected_seq7_mm_gh_dh),
]

# ===== Expected returns and advantages: Sequence 8 ===================================================================

# --- Multi-agent, single-value -------------------------------------------------------------

expected_seq8_ms_g0_d0 = {
    "advantages": [
        [-1, 1],
        [6]
    ],
    "returns": [
        [0, 2],
        [8]
    ],
}

expected_seq8_ms_g0_d1 = {
    "advantages": [
        [1, 3],
        [6]
    ],
    "returns": [
        [2, 4],
        [8]
    ],
}

expected_seq8_ms_g1_d1 = {
    "advantages": [
        [7, 3],
        [6]
    ],
    "returns": [
        [8, 4],
        [8]
    ],
}

expected_seq8_ms_gh_dh = {
    "advantages": [
        [2.4, 2],
        [6]
    ],
    "returns": [
        [3.4, 3],
        [8]
    ],
}

# --- Multi-agent, multi-value -------------------------------------------------------------

expected_seq8_mm_g0_d0 = {
    "advantages": [
        [-2, 0],
        [8]
    ],
    "returns": [
        [0, 2],
        [8]
    ],
}

expected_seq8_mm_g0_d1 = {
    "advantages": [
        [-2, 4],
        [8]
    ],
    "returns": [
        [0, 6],
        [8]
    ],
}

expected_seq8_mm_g1_d1 = {
    "advantages": [
        [6, 4],
        [8]
    ],
    "returns": [
        [8, 6],
        [8]
    ],
}

expected_seq8_mm_gh_dh = {
    "advantages": [
        [1.2, 2],
        [8]
    ],
    "returns": [
        [3.2, 4],
        [8]
    ],
}

# --- Settings for Sequence 8 -------------------------------------------------------------

settings_seq8 = [
    # single-agent, single-value is the same as in sequence 6
    # multi-agent, single-value
    (seq8_disappearance_2nd, func_args_ms, comp_args_g0_d0, expected_seq8_ms_g0_d0),
    (seq8_disappearance_2nd, func_args_ms, comp_args_g0_d1, expected_seq8_ms_g0_d1),
    (seq8_disappearance_2nd, func_args_ms, comp_args_g1_d1, expected_seq8_ms_g1_d1),
    (seq8_disappearance_2nd, func_args_ms, comp_args_gh_dh, expected_seq8_ms_gh_dh),
    # multi-agent, multi-value
    (seq8_disappearance_2nd, func_args_mm, comp_args_g0_d0, expected_seq8_mm_g0_d0),
    (seq8_disappearance_2nd, func_args_mm, comp_args_g0_d1, expected_seq8_mm_g0_d1),
    (seq8_disappearance_2nd, func_args_mm, comp_args_g1_d1, expected_seq8_mm_g1_d1),
    (seq8_disappearance_2nd, func_args_mm, comp_args_gh_dh, expected_seq8_mm_gh_dh),
]

# ===== Expected returns and advantages: Sequence 9 ===================================================================

# --- Multi-agent, single-value -------------------------------------------------------------


expected_seq9_ms_g0_d0 = {
    "advantages": [
        [-1],
        [6, 2]
    ],
    "returns": [
        [0],
        [8, 4]
    ],
}

expected_seq9_ms_g0_d1 = {
    "advantages": [
        [1],
        [6, 2]
    ],
    "returns": [
        [2],
        [8, 4]
    ],
}

expected_seq9_ms_g1_d1 = {
    "advantages": [
        [7],
        [6, 2]
    ],
    "returns": [
        [8],
        [8, 4]
    ],
}

expected_seq9_ms_gh_dh = {
    "advantages": [
        [2.4],
        [6, 2]
    ],
    "returns": [
        [3.4],
        [8, 4]
    ],
}

# --- Multi-agent, multi-value -------------------------------------------------------------

expected_seq9_mm_g0_d0 = {
    "advantages": [
        [-2],
        [8, 4]
    ],
    "returns": [
        [0],
        [8, 4]
    ],
}

expected_seq9_mm_g0_d1 = {
    "advantages": [
        [-2],
        [8, 4]
    ],
    "returns": [
        [0],
        [8, 4]
    ],
}

expected_seq9_mm_g1_d1 = {
    "advantages": [
        [6],
        [8, 4]
    ],
    "returns": [
        [8],
        [8, 4]
    ],
}

expected_seq9_mm_gh_dh = {
    "advantages": [
        [1.2],
        [8, 4]
    ],
    "returns": [
        [3.2],
        [8, 4]
    ],
}

# --- Settings for Sequence 9 -------------------------------------------------------------

settings_seq9 = [
    # single-agent, single-value is the same as in sequence 6
    # multi-agent, single-value
    (seq9_gone_and_return, func_args_ms, comp_args_g0_d0, expected_seq9_ms_g0_d0),
    (seq9_gone_and_return, func_args_ms, comp_args_g0_d1, expected_seq9_ms_g0_d1),
    (seq9_gone_and_return, func_args_ms, comp_args_g1_d1, expected_seq9_ms_g1_d1),
    (seq9_gone_and_return, func_args_ms, comp_args_gh_dh, expected_seq9_ms_gh_dh),
    # multi-agent, multi-value
    (seq9_gone_and_return, func_args_mm, comp_args_g0_d0, expected_seq9_mm_g0_d0),
    (seq9_gone_and_return, func_args_mm, comp_args_g0_d1, expected_seq9_mm_g0_d1),
    (seq9_gone_and_return, func_args_mm, comp_args_g1_d1, expected_seq9_mm_g1_d1),
    (seq9_gone_and_return, func_args_mm, comp_args_gh_dh, expected_seq9_mm_gh_dh),
]

# ===== Expected returns and advantages: Sequence 10 ==================================================================


# --- Single-agent, single-value -------------------------------------------------------------

expected_seq10_ss_g0_d0 = {
    "advantages": [
        [0],
        [2],
        [0],
        [2]
    ],
    "returns": [
        [1],
        [4],
        [1],
        [4]
    ],
}

expected_seq10_ss_g0_d1 = {
    "advantages": [
        [2],
        [2],
        [2],
        [2]
    ],
    "returns": [
        [3],
        [4],
        [3],
        [4]
    ],
}

expected_seq10_ss_g1_d1 = {
    "advantages": [
        [4],
        [2],
        [4],
        [2]
    ],
    "returns": [
        [5],
        [4],
        [5],
        [4]
    ],
}

expected_seq10_ss_gh_dh = {
    "advantages": [
        [1.8],
        [2],
        [1.8],
        [2]
    ],
    "returns": [
        [2.8],
        [4],
        [2.8],
        [4]
    ],
}

# --- Multi-agent, single-value -------------------------------------------------------------

expected_seq10_ms_g0_d0 = {
    "advantages": [
        [-1, 1],
        [6],
        [-1, 1, 2],
        [6, 2, 2]
    ],
    "returns": [
        [0, 2],
        [8],
        [0, 2, 3],
        [8, 4, 4]
    ],
}

expected_seq10_ms_g0_d1 = {
    "advantages": [
        [1, 3],
        [6],
        [1, 3, 4],
        [6, 2, 2]
    ],
    "returns": [
        [2, 4],
        [8],
        [2, 4, 5],
        [8, 4, 4]
    ],
}

expected_seq10_ms_g1_d1 = {
    "advantages": [
        [7, 3],
        [6],
        [7, 5, 6],
        [6, 2, 2]
    ],
    "returns": [
        [8, 4],
        [8],
        [8, 6, 7],
        [8, 4, 4]
    ],
}

expected_seq10_ms_gh_dh = {
    "advantages": [
        [2.4, 2],
        [6],
        [2.4, 2.8, 3.8],
        [6, 2, 2]
    ],
    "returns": [
        [3.4, 3],
        [8],
        [3.4, 3.8, 4.8],
        [8, 4, 4]
    ],
}

# --- Multi-agent, multi-value -------------------------------------------------------------

expected_seq10_mm_g0_d0 = {
    "advantages": [
        [-2, 0],
        [8],
        [-2, 0, 3],
        [8, 0, 5]
    ],
    "returns": [
        [0, 2],
        [8],
        [0, 2, 3],
        [8, 4, 4]
    ],
}

expected_seq10_mm_g0_d1 = {
    "advantages": [
        [-2, 4],
        [8],
        [-2, 4, 2],
        [8, 0, 5]
    ],
    "returns": [
        [0, 6],
        [8],
        [0, 6, 2],
        [8, 4, 4]
    ],
}

expected_seq10_mm_g1_d1 = {
    "advantages": [
        [6, 4],
        [8],
        [6, 4, 7],
        [8, 0, 5]
    ],
    "returns": [
        [8, 6],
        [8],
        [8, 6, 7],
        [8, 4, 4]
    ],
}

expected_seq10_mm_gh_dh = {
    "advantages": [
        [1.2, 2],
        [8],
        [1.2, 2, 4.5],
        [8, 0, 5]
    ],
    "returns": [
        [3.2, 4],
        [8],
        [3.2, 4, 4.5],
        [8, 4, 4]
    ],
}

# --- Settings for Sequence 10 -------------------------------------------------------------

settings_seq10 = [
    # single-agent, single-value
    (seq10_two_episodes, func_args_ss, comp_args_g0_d0, expected_seq10_ss_g0_d0),
    (seq10_two_episodes, func_args_ss, comp_args_g0_d1, expected_seq10_ss_g0_d1),
    (seq10_two_episodes, func_args_ss, comp_args_g1_d1, expected_seq10_ss_g1_d1),
    (seq10_two_episodes, func_args_ss, comp_args_gh_dh, expected_seq10_ss_gh_dh),
    # multi-agent, single-value
    (seq10_two_episodes, func_args_ms, comp_args_g0_d0, expected_seq10_ms_g0_d0),
    (seq10_two_episodes, func_args_ms, comp_args_g0_d1, expected_seq10_ms_g0_d1),
    (seq10_two_episodes, func_args_ms, comp_args_g1_d1, expected_seq10_ms_g1_d1),
    (seq10_two_episodes, func_args_ms, comp_args_gh_dh, expected_seq10_ms_gh_dh),
    # multi-agent, multi-value
    (seq10_two_episodes, func_args_mm, comp_args_g0_d0, expected_seq10_mm_g0_d0),
    (seq10_two_episodes, func_args_mm, comp_args_g0_d1, expected_seq10_mm_g0_d1),
    (seq10_two_episodes, func_args_mm, comp_args_gh_dh, expected_seq10_mm_gh_dh),
    (seq10_two_episodes, func_args_mm, comp_args_g1_d1, expected_seq10_mm_g1_d1),
]

# =====================================================================================================================

all_settings = (settings_seq1 +
                settings_seq2 +
                settings_seq3 +
                settings_seq5 +
                settings_seq6 +
                settings_seq7 +
                settings_seq8 +
                settings_seq9 +
                settings_seq10
                )

@pytest.mark.parametrize("setting", all_settings)
def test_ret_adv_computation(setting):

    (step_info, final_info), func_args, comp_args, expected = setting

    actor_loc = func_args["actor_loc"]
    value_loc = func_args["value_loc"]

    # assemble buffer data (T steps), simulating the observer filtering done by obs.get_{actor/value}_input().
    values, value_observers, rewards, reward_observers, dones = [], [], [], [], []
    for step in step_info:
        birdseye_observer_index = step["observers"].index(BIRDSEYE_OBSERVER_ID)
        if value_loc == "single":
            relevant_observers = [BIRDSEYE_OBSERVER_ID]
            relevant_values = [step["values"][birdseye_observer_index]]
        else:
            relevant_observers = [i for i in step["observers"]
                                        if i != BIRDSEYE_OBSERVER_ID]
            relevant_values = [step["values"][i] for i in range(len(step["observers"]))
                               if i != birdseye_observer_index]
        value_observers.append(torch.tensor(relevant_observers, dtype=torch.long))
        values.append(torch.tensor(relevant_values, dtype=torch.float))

        birdseye_next_observer_index = step["next_observers"].index(BIRDSEYE_OBSERVER_ID)
        if actor_loc == "single":
            relevant_next_observers = [BIRDSEYE_OBSERVER_ID]
            relevant_rewards = [step["rewards"][birdseye_next_observer_index]]
        else:
            relevant_next_observers = [i for i in step["next_observers"]
                                         if i != BIRDSEYE_OBSERVER_ID]
            relevant_rewards = [step["rewards"][i] for i in range(len(step["next_observers"]))
                                if i != birdseye_next_observer_index]
        reward_observers.append(torch.tensor(relevant_next_observers, dtype=torch.long))
        rewards.append(torch.tensor(relevant_rewards, dtype=torch.float))

        dones.append(torch.tensor(step["done"], dtype=torch.float))

    last_observers = step_info[-1]["next_observers"]
    birdseye_final_observer_index = last_observers.index(BIRDSEYE_OBSERVER_ID)
    if value_loc == "single":
        relevant_last_observers = [BIRDSEYE_OBSERVER_ID]
        relevant_last_values = [final_info["last_values"][birdseye_final_observer_index]]
    else:
        relevant_last_observers = [i for i in last_observers  if i != BIRDSEYE_OBSERVER_ID]
        relevant_last_values = [final_info["last_values"][i] for i in range(len(last_observers))
                                if i != birdseye_final_observer_index]
    last_value_observers = torch.tensor(relevant_last_observers, dtype=torch.long)
    last_values = torch.tensor(relevant_last_values, dtype=torch.float)
    dones = torch.stack(dones)

    # assemble expected returns and advantages
    expected_returns = expected["returns"]
    expected_advantages = expected["advantages"]
    if expected_returns is NotImplemented or expected_advantages is NotImplemented:
        # Also skip if expected returns or advantages are not defined
        pytest.skip("Expected returns or advantages are not defined")
    expected_returns = [torch.tensor(ret, dtype=torch.double) for ret in expected_returns]
    expected_advantages = [torch.tensor(adv, dtype=torch.double) for adv in expected_advantages]

    actual_returns, actual_advantages = compute_returns_and_advantages(
        values=values,  # length T
        value_observers=value_observers,  # length T+1
        rewards=rewards,  # length T
        reward_observers=reward_observers,
        #last_values=last_values,
        #last_value_observers=last_value_observers,
        dones=dones,  # tensor of shape (T,)
        device=torch.device("cpu"),
        actor_loc=actor_loc,
        value_loc=value_loc,
        gae_lambda=comp_args['gae_lambda'],
        discount_factor=comp_args['discount_factor'],
        debug=True
    )

    # check list lengths
    assert len(actual_returns) == len(actual_advantages), "Returns and advantages should have the same length"

    for t, (expected_ret_t, actual_ret_t, expected_adv_t, actual_adv_t) \
            in enumerate(zip(expected_returns, actual_returns, expected_advantages, actual_advantages)):

        # check dtypes
        assert actual_adv_t.dtype == expected_adv_t.dtype, \
            f"Advantage dtype mismatch in step {t}: expected {expected_adv_t.dtype}, got {actual_adv_t.dtype}"
        assert actual_ret_t.dtype == expected_ret_t.dtype, \
            f"Return dtype mismatch in step {t}: expected {expected_ret_t.dtype}, got {actual_ret_t.dtype}"

        # check shapes
        assert actual_adv_t.shape == expected_adv_t.shape, \
            f"Advantage shape mismatch in step {t}: expected {expected_adv_t.shape}, got {actual_adv_t.shape}"
        assert actual_ret_t.shape == expected_ret_t.shape, \
            f"Return shape mismatch in step {t}: expected {expected_ret_t.shape}, got {actual_ret_t.shape}"

        # check contents
        assert torch.allclose(actual_adv_t, expected_adv_t, atol=1e-6), \
            f"Advantage mismatch in step {t}: expected {expected_adv_t}, got {actual_adv_t}"
        assert torch.allclose(actual_ret_t, expected_ret_t, atol=1e-6), \
            f"Return mismatch in step {t}: expected {expected_ret_t}, got {actual_ret_t}"
