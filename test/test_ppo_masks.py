import sys
sys.path.append(".")

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
import pytest

from ._mock_obs import *
from src.algorithm.trainable import get_trainable_algorithm
from src.trainer.ppo import get_ppo_trainer


# === func args =============================================================

sb_func_args = {"actor_loc": "single", "actor_input": "birdseye"}
sc_func_args = {"actor_loc": "single", "actor_input": "central"}
mb_func_args = {"actor_loc": "multi", "actor_input": "birdseye"}
mc_func_args = {"actor_loc": "multi", "actor_input": "central"}
ml_func_args = {"actor_loc": "multi", "actor_input": "local"}

func_arg_list_2args = [sb_func_args, sc_func_args, mb_func_args, mc_func_args, ml_func_args]

# combine dicts of func_arg_list_2args with additional entry "algo_config" == "fieldlines" or "mslim"
func_arg_list_3args = []
for func_args in func_arg_list_2args:
    func_arg_list_3args.append({**func_args, **{"algo_config": "fieldlines"}})
    func_arg_list_3args.append({**func_args, **{"algo_config": "mslim"}})

# combine dicts of func_arg_list_3_args with additional entry "ppo_fl_style" == "tmlr24" or "new" for fieldlines only
func_arg_list_4args = []
for func_args in func_arg_list_3args:
    if func_args["algo_config"] == "fieldlines":
        func_arg_list_4args.append({**func_args, **{"ppo_fl_style": "new"}})
        func_arg_list_4args.append({**func_args, **{"ppo_fl_style": "tmlr24"}})
    else:
        func_arg_list_4args.append(func_args)

# === get_logprob_mask (3arg) =======================================================

# expected outputs for get_logprob_mask() on batch1 = [obs1, obs2]
glm_sbn_expected1 = torch.ones(obs1bN ** 2 + obs2bN ** 2, dtype=torch.bool)
glm_sbl_expected1 = torch.ones(obs1bE + obs2bE, dtype=torch.bool)
glm_scn_expected1 = torch.ones(obs1cN ** 2 + obs2cN ** 2, dtype=torch.bool)
glm_scl_expected1 = torch.ones(obs1cE + obs2cE, dtype=torch.bool)
glm_mbn_expected1 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1 - observer 0, birdseye graph, N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1 - observer 1, birdseye graph, N=2
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs2 - observer 0, birdseye graph, N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2 - observer 1, birdseye graph, N=2
))
glm_mbl_expected1 = torch.cat((
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 1, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs2 - observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs2 - observer 1, birdseye graph, edge_index[0] = [0, 1]
))  # logprob mask for multiagent link-weight policies is full mask!
glm_mcn_expected1 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1 - observer 0, central graph (node 0), N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1 - observer 1, central graph (node 0), N=2
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2 - observer 0, central graph (node 0), N=3
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # obs2 - observer 1, central graph (node 0), N=3
))
glm_mcl_expected1 = torch.cat((
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 0, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 1, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs2 - observer 0, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs2 - observer 1, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
))  # logprob mask for multiagent link-weight policies is full mask!
glm_mln_expected1 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1 - observer 0's graph, N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1 - observer 1's graph, N=2
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2 - observer 0's graph, N=3
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2 - observer 1's graph, N=2
))
glm_mll_expected1 = torch.cat((
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 0's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 1's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs2 - observer 0's graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1], dtype=torch.bool),  # obs2 - observer 1's graph, edge_index[0] = [0, 1]
))  # logprob mask for multiagent link-weight policies is full mask!

get_logprob_mask_expected1 = [
    glm_sbn_expected1,
    glm_sbl_expected1,
    glm_scn_expected1,
    glm_scl_expected1,
    glm_mbn_expected1,
    glm_mbl_expected1,
    glm_mcn_expected1,
    glm_mcl_expected1,
    glm_mln_expected1,
    glm_mll_expected1,
]

get_logprob_mask_settings_batch1 = ([("_get_logprob_mask", obs_batch1, expected, func_args) for (expected, func_args)
                                     in zip(get_logprob_mask_expected1, func_arg_list_3args)])

# expected outputs for get_logprob_mask() on batch2
glm_sbn_expected2 = torch.ones(obs2bN ** 2 + obs1bN ** 2, dtype=torch.bool)
glm_sbl_expected2 = torch.ones(obs2bE + obs1bE, dtype=torch.bool)
glm_scn_expected2 = torch.ones(obs2cN ** 2 + obs1cN ** 2, dtype=torch.bool)
glm_scl_expected2 = torch.ones(obs2cE + obs1cE, dtype=torch.bool)
glm_mbn_expected2 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs2 - observer 0, birdseye graph, N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2 - observer 1, birdseye graph, N=2
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1 - observer 0, birdseye graph, N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1 - observer 1, birdseye graph, N=2
))
glm_mbl_expected2 = torch.cat((
    torch.tensor([1, 1], dtype=torch.bool),  # obs2 - observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs2 - observer 1, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 1, birdseye graph, edge_index[0] = [0, 1]
))  # logprob mask for multiagent link-weight policies is full mask!
glm_mcn_expected2 = torch.cat((
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2 - observer 0, central graph (node 0), N=3
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # obs2 - observer 1, central graph (node 0), N=3
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1 - observer 0, central graph (node 0), N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1 - observer 1, central graph (node 0), N=2
))
glm_mcl_expected2 = torch.cat((
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs2 - observer 0, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs2 - observer 1, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 0, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 1, central graph (node 0), edge_index[0] = [0, 1]
))  # logprob mask for multiagent link-weight policies is full mask!
glm_mln_expected2 = torch.cat((
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2 - observer 0's graph, N=3
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2 - observer 1's graph, N=2
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1 - observer 0's graph, N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1 - observer 1's graph, N=2
))
glm_mll_expected2 = torch.cat((
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs2 - observer 0's graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1], dtype=torch.bool),  # obs2 - observer 1's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 0's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 1's graph, edge_index[0] = [0, 1]
))  # logprob mask for multiagent link-weight policies is full mask!

get_logprob_mask_expected2 = [
    glm_sbn_expected2,
    glm_sbl_expected2,
    glm_scn_expected2,
    glm_scl_expected2,
    glm_mbn_expected2,
    glm_mbl_expected2,
    glm_mcn_expected2,
    glm_mcl_expected2,
    glm_mln_expected2,
    glm_mll_expected2,
]

get_logprob_mask_settings_batch2 = ([("_get_logprob_mask", obs_batch2, expected, func_args) for (expected, func_args)
                                     in zip(get_logprob_mask_expected2, func_arg_list_3args)])

# expected outputs for get_logprob_mask() on batch3
glm_sbn_expected3 = torch.ones(obs1bN ** 2 + obs2bN ** 2 + obs1bN ** 2, dtype=torch.bool)
glm_sbl_expected3 = torch.ones(obs1bE + obs2bE + obs1bE, dtype=torch.bool)
glm_scn_expected3 = torch.ones(obs1cN ** 2 + obs2cN ** 2 + obs1cN ** 2, dtype=torch.bool)
glm_scl_expected3 = torch.ones(obs1cE + obs2cE + obs1cE, dtype=torch.bool)
glm_mbn_expected3 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1 - observer 0, birdseye graph, N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1 - observer 1, birdseye graph, N=2
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs2 - observer 0, birdseye graph, N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2 - observer 1, birdseye graph, N=2
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1 - observer 0, birdseye graph, N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1 - observer 1, birdseye graph, N=2
))
glm_mbl_expected3 = torch.cat((
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 1, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs2 - observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs2 - observer 1, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 1, birdseye graph, edge_index[0] = [0, 1]
))  # logprob mask for multiagent link-weight policies is full mask!
glm_mcn_expected3 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1 - observer 0, central graph (node 0), N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1 - observer 1, central graph (node 0), N=2
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2 - observer 0, central graph (node 0), N=3
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # obs2 - observer 1, central graph (node 0), N=3
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1 - observer 0, central graph (node 0), N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1 - observer 1, central graph (node 0), N=2
))
glm_mcl_expected3 = torch.cat((
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 0, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 1, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs2 - observer 0, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs2 - observer 1, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 0, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 1, central graph (node 0), edge_index[0] = [0, 1]
))  # logprob mask for multiagent link-weight policies is full mask!
glm_mln_expected3 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1 - observer 0's graph, N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1 - observer 1's graph, N=2
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2 - observer 0's graph, N=3
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2 - observer 1's graph, N=2
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1 - observer 0's graph, N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1 - observer 1's graph, N=2
))
glm_mll_expected3 = torch.cat((
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 0's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 1's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs2 - observer 0's graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1], dtype=torch.bool),  # obs2 - observer 1's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 0's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 1's graph, edge_index[0] = [0, 1]
))  # logprob mask for multiagent link-weight policies is full mask!

get_logprob_mask_expected3 = [
    glm_sbn_expected3,
    glm_sbl_expected3,
    glm_scn_expected3,
    glm_scl_expected3,
    glm_mbn_expected3,
    glm_mbl_expected3,
    glm_mcn_expected3,
    glm_mcl_expected3,
    glm_mln_expected3,
    glm_mll_expected3,
]

get_logprob_mask_settings_batch3 = ([("_get_logprob_mask", obs_batch3, expected, func_args) for (expected, func_args)
                                     in zip(get_logprob_mask_expected3, func_arg_list_3args)])

# expected outputs for get_logprob_mask() on batch4
glm_sbn_expected4 = torch.ones(obs1bN ** 2 + obs2bN ** 2 + obs3bN ** 2, dtype=torch.bool)
glm_sbl_expected4 = torch.ones(obs1bE + obs2bE + obs3bE, dtype=torch.bool)
glm_scn_expected4 = torch.ones(obs1cN ** 2 + obs2cN ** 2 + obs3cN ** 2, dtype=torch.bool)
glm_scl_expected4 = torch.ones(obs1cE + obs2cE + obs3cE, dtype=torch.bool)
glm_mbn_expected4 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1 - observer 0, birdseye graph, N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1 - observer 1, birdseye graph, N=2
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs2 - observer 0, birdseye graph, N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2 - observer 1, birdseye graph, N=2
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs3 - observer 0, birdseye graph, N=3
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # obs3 - observer 1, birdseye graph, N=3
    torch.tensor([0, 0, 0,
                  0, 0, 0,
                  1, 1, 1], dtype=torch.bool),  # obs3 - observer 1, birdseye graph, N=3
))
glm_mbl_expected4 = torch.cat((
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 1, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs2 - observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs2 - observer 1, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs3 - observer 0, birdseye graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs3 - observer 1, birdseye graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs3 - observer 2, birdseye graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
))  # logprob mask for multiagent link-weight policies is full mask!
glm_mcn_expected4 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1 - observer 0, central graph (node 0), N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1 - observer 1, central graph (node 0), N=2
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2 - observer 0, central graph (node 0), N=3
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # obs2 - observer 1, central graph (node 0), N=3
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs3 - observer 0, central graph (node 0), N=3
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # obs3 - observer 1, central graph (node 0), N=3
    torch.tensor([0, 0, 0,
                  0, 0, 0,
                  1, 1, 1], dtype=torch.bool),  # obs3 - observer 2, central graph (node 0), N=3
))
glm_mcl_expected4 = torch.cat((
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 0, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 1, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs2 - observer 0, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs2 - observer 1, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs3 - observer 0, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs3 - observer 1, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs3 - observer 2, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
))  # logprob mask for multiagent link-weight policies is full mask!
glm_mln_expected4 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1 - observer 0's graph, N=2
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1 - observer 1's graph, N=2
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2 - observer 0's graph, N=3
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2 - observer 1's graph, N=2
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs3 - observer 0's graph, N=3
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # obs3 - observer 1's graph, N=3
    torch.tensor([0, 0, 0,
                  0, 0, 0,
                  1, 1, 1], dtype=torch.bool),  # obs3 - observer 2's graph, N=3
))
glm_mll_expected4 = torch.cat((
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 0's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1], dtype=torch.bool),  # obs1 - observer 1's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs2 - observer 0's graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1], dtype=torch.bool),  # obs2 - observer 1's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs3 - observer 0's graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs3 - observer 1's graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool),  # obs3 - observer 2's graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
))  # logprob mask for multiagent link-weight policies is full mask!

get_logprob_mask_expected4 = [
    glm_sbn_expected4,
    glm_sbl_expected4,
    glm_scn_expected4,
    glm_scl_expected4,
    glm_mbn_expected4,
    glm_mbl_expected4,
    glm_mcn_expected4,
    glm_mcl_expected4,
    glm_mln_expected4,
    glm_mll_expected4,
]

get_logprob_mask_settings_batch4 = ([("_get_logprob_mask", obs_batch4, expected, func_args) for (expected, func_args)
                                     in zip(get_logprob_mask_expected4, func_arg_list_3args)])

get_logprob_mask_settings = (get_logprob_mask_settings_batch1
                             + get_logprob_mask_settings_batch2
                             + get_logprob_mask_settings_batch3
                             + get_logprob_mask_settings_batch4)

# === get_logprob_graph_idx (3arg) =======================================================

# expected outputs for get_logprob_graph_idx() on batch1
glgi_sbn_expected1 = torch.cat((
    torch.tensor([0] * obs1bN ** 2, dtype=torch.long),  # obs1 - birdseye graph, N=2
    torch.tensor([1] * obs2bN ** 2, dtype=torch.long),  # obs2 - birdseye graph, N=3
))
glgi_sbl_expected1 = torch.cat((
    torch.tensor([0] * obs1bE, dtype=torch.long),  # obs1 - birdseye graph, E=2
    torch.tensor([1] * obs2bE, dtype=torch.long),  # obs2 - birdseye graph, E=2
))
glgi_scn_expected1 = torch.cat((
    torch.tensor([0] * obs1cN ** 2, dtype=torch.long),  # obs1 - central graph (node 0), N=2
    torch.tensor([1] * obs2cN ** 2, dtype=torch.long),  # obs2 - central graph (node 0), N=3
))
glgi_scl_expected1 = torch.cat((
    torch.tensor([0] * obs1cE, dtype=torch.long),  # obs1 - central graph (node 0), E=2
    torch.tensor([1] * obs2cE, dtype=torch.long),  # obs2 - central graph (node 0), E=6
))
glgi_mbn_expected1 = torch.cat((
    torch.tensor([0] * obs1bN ** 2, dtype=torch.long),  # obs1 - observer 0, birdseye graph, N=2
    torch.tensor([1] * obs1bN ** 2, dtype=torch.long),  # obs1 - observer 1, birdseye graph, N=2
    torch.tensor([2] * obs2bN ** 2, dtype=torch.long),  # obs2 - observer 0, birdseye graph, N=2
    torch.tensor([3] * obs2bN ** 2, dtype=torch.long),  # obs2 - observer 1, birdseye graph, N=2
))
glgi_mbl_expected1 = torch.cat((
    torch.tensor([0] * obs1bE, dtype=torch.long),  # obs1 - observer 0, birdseye graph, E=2
    torch.tensor([1] * obs1bE, dtype=torch.long),  # obs1 - observer 1, birdseye graph, E=2
    torch.tensor([2] * obs2bE, dtype=torch.long),  # obs2 - observer 0, birdseye graph, E=2
    torch.tensor([3] * obs2bE, dtype=torch.long),  # obs2 - observer 1, birdseye graph, E=2
))
glgi_mcn_expected1 = torch.cat((
    torch.tensor([0] * obs1cN ** 2, dtype=torch.long),  # obs1 - observer 0, central graph (node 0), N=2
    torch.tensor([1] * obs1cN ** 2, dtype=torch.long),  # obs1 - observer 1, central graph (node 0), N=2
    torch.tensor([2] * obs2cN ** 2, dtype=torch.long),  # obs2 - observer 0, central graph (node 0), N=3
    torch.tensor([3] * obs2cN ** 2, dtype=torch.long),  # obs2 - observer 1, central graph (node 0), N=3
))
glgi_mcl_expected1 = torch.cat((
    torch.tensor([0] * obs1cE, dtype=torch.long),  # obs1 - observer 0, central graph (node 0), E=2
    torch.tensor([1] * obs1cE, dtype=torch.long),  # obs1 - observer 1, central graph (node 0), E=2
    torch.tensor([2] * obs2cE, dtype=torch.long),  # obs2 - observer 0, central graph (node 0), E=6
    torch.tensor([3] * obs2cE, dtype=torch.long),  # obs2 - observer 1, central graph (node 0), E=6
))
glgi_mln_expected1 = torch.cat((
    torch.tensor([0] * obs1n0N ** 2, dtype=torch.long),  # obs1 - observer 0's graph, N=2
    torch.tensor([1] * obs1n1N ** 2, dtype=torch.long),  # obs1 - observer 1's graph, N=2
    torch.tensor([2] * obs2n0N ** 2, dtype=torch.long),  # obs2 - observer 0's graph, N=3
    torch.tensor([3] * obs2n1N ** 2, dtype=torch.long),  # obs2 - observer 1's graph, N=2
))
glgi_mll_expected1 = torch.cat((
    torch.tensor([0] * obs1n0E, dtype=torch.long),  # obs1 - observer 0's graph, E=2
    torch.tensor([1] * obs1n1E, dtype=torch.long),  # obs1 - observer 1's graph, E=2
    torch.tensor([2] * obs2n0E, dtype=torch.long),  # obs2 - observer 0's graph, E=6
    torch.tensor([3] * obs2n1E, dtype=torch.long),  # obs2 - observer 1's graph, E=2
))

get_logprob_graph_idx_expected1 = [
    glgi_sbn_expected1,
    glgi_sbl_expected1,
    glgi_scn_expected1,
    glgi_scl_expected1,
    glgi_mbn_expected1,
    glgi_mbl_expected1,
    glgi_mcn_expected1,
    glgi_mcl_expected1,
    glgi_mln_expected1,
    glgi_mll_expected1,
]

get_logprob_graph_idx_settings_batch1 = ([("_get_logprob_graph_idx", obs_batch1, expected, func_args) for (expected, func_args)
                                            in zip(get_logprob_graph_idx_expected1, func_arg_list_3args)])

# expected outputs for get_logprob_graph_idx() on batch2
glgi_sbn_expected2 = torch.cat((
    torch.tensor([0] * obs2bN ** 2, dtype=torch.long),  # obs2 - birdseye graph, N=2
    torch.tensor([1] * obs1bN ** 2, dtype=torch.long),  # obs1 - birdseye graph, N=2
))
glgi_sbl_expected2 = torch.cat((
    torch.tensor([0] * obs2bE, dtype=torch.long),  # obs2 - birdseye graph, E=2
    torch.tensor([1] * obs1bE, dtype=torch.long),  # obs1 - birdseye graph, E=2
))
glgi_scn_expected2 = torch.cat((
    torch.tensor([0] * obs2cN ** 2, dtype=torch.long),  # obs2 - central graph (node 0), N=3
    torch.tensor([1] * obs1cN ** 2, dtype=torch.long),  # obs1 - central graph (node 0), N=2
))
glgi_scl_expected2 = torch.cat((
    torch.tensor([0] * obs2cE, dtype=torch.long),  # obs2 - central graph (node 0), E=6
    torch.tensor([1] * obs1cE, dtype=torch.long),  # obs1 - central graph (node 0), E=2
))
glgi_mbn_expected2 = torch.cat((
    torch.tensor([0] * obs2bN ** 2, dtype=torch.long),  # obs2 - observer 0, birdseye graph, N=2
    torch.tensor([1] * obs2bN ** 2, dtype=torch.long),  # obs2 - observer 1, birdseye graph, N=2
    torch.tensor([2] * obs1bN ** 2, dtype=torch.long),  # obs1 - observer 0, birdseye graph, N=2
    torch.tensor([3] * obs1bN ** 2, dtype=torch.long),  # obs1 - observer 1, birdseye graph, N=2
))
glgi_mbl_expected2 = torch.cat((
    torch.tensor([0] * obs2bE, dtype=torch.long),  # obs2 - observer 0, birdseye graph, E=2
    torch.tensor([1] * obs2bE, dtype=torch.long),  # obs2 - observer 1, birdseye graph, E=2
    torch.tensor([2] * obs1bE, dtype=torch.long),  # obs1 - observer 0, birdseye graph, E=2
    torch.tensor([3] * obs1bE, dtype=torch.long),  # obs1 - observer 1, birdseye graph, E=2
))
glgi_mcn_expected2 = torch.cat((
    torch.tensor([0] * obs2cN ** 2, dtype=torch.long),  # obs2 - observer 0, central graph (node 0), N=3
    torch.tensor([1] * obs2cN ** 2, dtype=torch.long),  # obs2 - observer 1, central graph (node 0), N=3
    torch.tensor([2] * obs1cN ** 2, dtype=torch.long),  # obs1 - observer 0, central graph (node 0), N=2
    torch.tensor([3] * obs1cN ** 2, dtype=torch.long),  # obs1 - observer 1, central graph (node 0), N=2
))
glgi_mcl_expected2 = torch.cat((
    torch.tensor([0] * obs2cE, dtype=torch.long),  # obs2 - observer 0, central graph (node 0), E=6
    torch.tensor([1] * obs2cE, dtype=torch.long),  # obs2 - observer 1, central graph (node 0), E=6
    torch.tensor([2] * obs1cE, dtype=torch.long),  # obs1 - observer 0, central graph (node 0), E=2
    torch.tensor([3] * obs1cE, dtype=torch.long),  # obs1 - observer 1, central graph (node 0), E=2
))
glgi_mln_expected2 = torch.cat((
    torch.tensor([0] * obs2n0N ** 2, dtype=torch.long),  # obs2 - observer 0's graph, N=3
    torch.tensor([1] * obs2n1N ** 2, dtype=torch.long),  # obs2 - observer 1's graph, N=2
    torch.tensor([2] * obs1n0N ** 2, dtype=torch.long),  # obs1 - observer 0's graph, N=2
    torch.tensor([3] * obs1n1N ** 2, dtype=torch.long),  # obs1 - observer 1's graph, N=2
))
glgi_mll_expected2 = torch.cat((
    torch.tensor([0] * obs2n0E, dtype=torch.long),  # obs2 - observer 0's graph, E=6
    torch.tensor([1] * obs2n1E, dtype=torch.long),  # obs2 - observer 1's graph, E=2
    torch.tensor([2] * obs1n0E, dtype=torch.long),  # obs1 - observer 0's graph, E=2
    torch.tensor([3] * obs1n1E, dtype=torch.long),  # obs1 - observer 1's graph, E=2
))

get_logprob_graph_idx_expected2 = [
    glgi_sbn_expected2,
    glgi_sbl_expected2,
    glgi_scn_expected2,
    glgi_scl_expected2,
    glgi_mbn_expected2,
    glgi_mbl_expected2,
    glgi_mcn_expected2,
    glgi_mcl_expected2,
    glgi_mln_expected2,
    glgi_mll_expected2,
]

get_logprob_graph_idx_settings_batch2 = ([("_get_logprob_graph_idx", obs_batch2, expected, func_args) for (expected, func_args)
                                          in zip(get_logprob_graph_idx_expected2, func_arg_list_3args)])

# expected outputs for get_logprob_graph_idx() on batch3
glgi_sbn_expected3 = torch.cat((
    torch.tensor([0] * obs1bN ** 2, dtype=torch.long),  # obs1 - birdseye graph, N=2
    torch.tensor([1] * obs2bN ** 2, dtype=torch.long),  # obs2 - birdseye graph, N=3
    torch.tensor([2] * obs1bN ** 2, dtype=torch.long),  # obs1 - birdseye graph, N=2
))
glgi_sbl_expected3 = torch.cat((
    torch.tensor([0] * obs1bE, dtype=torch.long),  # obs1 - birdseye graph, E=2
    torch.tensor([1] * obs2bE, dtype=torch.long),  # obs2 - birdseye graph, E=2
    torch.tensor([2] * obs1bE, dtype=torch.long),  # obs1 - birdseye graph, E=2
))
glgi_scn_expected3 = torch.cat((
    torch.tensor([0] * obs1cN ** 2, dtype=torch.long),  # obs1 - central graph (node 0), N=2
    torch.tensor([1] * obs2cN ** 2, dtype=torch.long),  # obs2 - central graph (node 0), N=3
    torch.tensor([2] * obs1cN ** 2, dtype=torch.long),  # obs1 - central graph (node 0), N=2
))
glgi_scl_expected3 = torch.cat((
    torch.tensor([0] * obs1cE, dtype=torch.long),  # obs1 - central graph (node 0), E=2
    torch.tensor([1] * obs2cE, dtype=torch.long),  # obs2 - central graph (node 0), E=6
    torch.tensor([2] * obs1cE, dtype=torch.long),  # obs1 - central graph (node 0), E=2
))
glgi_mbn_expected3 = torch.cat((
    torch.tensor([0] * obs1bN ** 2, dtype=torch.long),  # obs1 - observer 0, birdseye graph, N=2
    torch.tensor([1] * obs1bN ** 2, dtype=torch.long),  # obs1 - observer 1, birdseye graph, N=2
    torch.tensor([2] * obs2bN ** 2, dtype=torch.long),  # obs2 - observer 0, birdseye graph, N=2
    torch.tensor([3] * obs2bN ** 2, dtype=torch.long),  # obs2 - observer 1, birdseye graph, N=2
    torch.tensor([4] * obs1bN ** 2, dtype=torch.long),  # obs1 - observer 0, birdseye graph, N=2
    torch.tensor([5] * obs1bN ** 2, dtype=torch.long),  # obs1 - observer 1, birdseye graph, N=2
))
glgi_mbl_expected3 = torch.cat((
    torch.tensor([0] * obs1bE, dtype=torch.long),  # obs1 - observer 0, birdseye graph, E=2
    torch.tensor([1] * obs1bE, dtype=torch.long),  # obs1 - observer 1, birdseye graph, E=2
    torch.tensor([2] * obs2bE, dtype=torch.long),  # obs2 - observer 0, birdseye graph, E=2
    torch.tensor([3] * obs2bE, dtype=torch.long),  # obs2 - observer 1, birdseye graph, E=2
    torch.tensor([4] * obs1bE, dtype=torch.long),  # obs1 - observer 0, birdseye graph, E=2
    torch.tensor([5] * obs1bE, dtype=torch.long),  # obs1 - observer 1, birdseye graph, E=2
))
glgi_mcn_expected3 = torch.cat((
    torch.tensor([0] * obs1cN ** 2, dtype=torch.long),  # obs1 - observer 0, central graph (node 0), N=2
    torch.tensor([1] * obs1cN ** 2, dtype=torch.long),  # obs1 - observer 1, central graph (node 0), N=2
    torch.tensor([2] * obs2cN ** 2, dtype=torch.long),  # obs2 - observer 0, central graph (node 0), N=3
    torch.tensor([3] * obs2cN ** 2, dtype=torch.long),  # obs2 - observer 1, central graph (node 0), N=3
    torch.tensor([4] * obs1cN ** 2, dtype=torch.long),  # obs1 - observer 0, central graph (node 0), N=2
    torch.tensor([5] * obs1cN ** 2, dtype=torch.long),  # obs1 - observer 1, central graph (node 0), N=2
))
glgi_mcl_expected3 = torch.cat((
    torch.tensor([0] * obs1cE, dtype=torch.long),  # obs1 - observer 0, central graph (node 0), E=2
    torch.tensor([1] * obs1cE, dtype=torch.long),  # obs1 - observer 1, central graph (node 0), E=2
    torch.tensor([2] * obs2cE, dtype=torch.long),  # obs2 - observer 0, central graph (node 0), E=6
    torch.tensor([3] * obs2cE, dtype=torch.long),  # obs2 - observer 1, central graph (node 0), E=6
    torch.tensor([4] * obs1cE, dtype=torch.long),  # obs1 - observer 0, central graph (node 0), E=2
    torch.tensor([5] * obs1cE, dtype=torch.long),  # obs1 - observer 1, central graph (node 0), E=2
))
glgi_mln_expected3 = torch.cat((
    torch.tensor([0] * obs1n0N ** 2, dtype=torch.long),  # obs1 - observer 0's graph, N=2
    torch.tensor([1] * obs1n1N ** 2, dtype=torch.long),  # obs1 - observer 1's graph, N=2
    torch.tensor([2] * obs2n0N ** 2, dtype=torch.long),  # obs2 - observer 0's graph, N=3
    torch.tensor([3] * obs2n1N ** 2, dtype=torch.long),  # obs2 - observer 1's graph, N=2
    torch.tensor([4] * obs1n0N ** 2, dtype=torch.long),  # obs1 - observer 0's graph, N=2
    torch.tensor([5] * obs1n1N ** 2, dtype=torch.long),  # obs1 - observer 1's graph, N=2
))
glgi_mll_expected3 = torch.cat((
    torch.tensor([0] * obs1n0E, dtype=torch.long),  # obs1 - observer 0's graph, E=2
    torch.tensor([1] * obs1n1E, dtype=torch.long),  # obs1 - observer 1's graph, E=2
    torch.tensor([2] * obs2n0E, dtype=torch.long),  # obs2 - observer 0's graph, E=6
    torch.tensor([3] * obs2n1E, dtype=torch.long),  # obs2 - observer 1's graph, E=2
    torch.tensor([4] * obs1n0E, dtype=torch.long),  # obs1 - observer 0's graph, E=2
    torch.tensor([5] * obs1n1E, dtype=torch.long),  # obs1 - observer 1's graph, E=2
))

get_logprob_graph_idx_expected3 = [
    glgi_sbn_expected3,
    glgi_sbl_expected3,
    glgi_scn_expected3,
    glgi_scl_expected3,
    glgi_mbn_expected3,
    glgi_mbl_expected3,
    glgi_mcn_expected3,
    glgi_mcl_expected3,
    glgi_mln_expected3,
    glgi_mll_expected3,
]

get_logprob_graph_idx_settings_batch3 = ([("_get_logprob_graph_idx", obs_batch3, expected, func_args) for (expected, func_args)
                                          in zip(get_logprob_graph_idx_expected3, func_arg_list_3args)])

# expected outputs for get_logprob_graph_idx() on batch4
glgi_sbn_expected4 = torch.cat((
    torch.tensor([0] * obs1bN ** 2, dtype=torch.long),  # obs1 - birdseye graph, N=2
    torch.tensor([1] * obs2bN ** 2, dtype=torch.long),  # obs2 - birdseye graph, N=2
    torch.tensor([2] * obs3bN ** 2, dtype=torch.long),  # obs3 - birdseye graph, N=3
))
glgi_sbl_expected4 = torch.cat((
    torch.tensor([0] * obs1bE, dtype=torch.long),  # obs1 - birdseye graph, E=2
    torch.tensor([1] * obs2bE, dtype=torch.long),  # obs2 - birdseye graph, E=2
    torch.tensor([2] * obs3bE, dtype=torch.long),  # obs3 - birdseye graph, E=6
))
glgi_scn_expected4 = torch.cat((
    torch.tensor([0] * obs1cN ** 2, dtype=torch.long),  # obs1 - central graph (node 0), N=2
    torch.tensor([1] * obs2cN ** 2, dtype=torch.long),  # obs2 - central graph (node 0), N=3
    torch.tensor([2] * obs3cN ** 2, dtype=torch.long),  # obs3 - central graph (node 0), N=3
))
glgi_scl_expected4 = torch.cat((
    torch.tensor([0] * obs1cE, dtype=torch.long),  # obs1 - central graph (node 0), E=2
    torch.tensor([1] * obs2cE, dtype=torch.long),  # obs2 - central graph (node 0), E=6
    torch.tensor([2] * obs3cE, dtype=torch.long),  # obs3 - central graph (node 0), E=6
))
glgi_mbn_expected4 = torch.cat((
    torch.tensor([0] * obs1bN ** 2, dtype=torch.long),  # obs1 - observer 0, birdseye graph, N=2
    torch.tensor([1] * obs1bN ** 2, dtype=torch.long),  # obs1 - observer 1, birdseye graph, N=2
    torch.tensor([2] * obs2bN ** 2, dtype=torch.long),  # obs2 - observer 0, birdseye graph, N=2
    torch.tensor([3] * obs2bN ** 2, dtype=torch.long),  # obs2 - observer 1, birdseye graph, N=2
    torch.tensor([4] * obs3bN ** 2, dtype=torch.long),  # obs3 - observer 0, birdseye graph, N=3
    torch.tensor([5] * obs3bN ** 2, dtype=torch.long),  # obs3 - observer 1, birdseye graph, N=3
    torch.tensor([6] * obs3bN ** 2, dtype=torch.long),  # obs3 - observer 2, birdseye graph, N=3
))
glgi_mbl_expected4 = torch.cat((
    torch.tensor([0] * obs1bE, dtype=torch.long),  # obs1 - observer 0, birdseye graph, E=2
    torch.tensor([1] * obs1bE, dtype=torch.long),  # obs1 - observer 1, birdseye graph, E=2
    torch.tensor([2] * obs2bE, dtype=torch.long),  # obs2 - observer 0, birdseye graph, E=2
    torch.tensor([3] * obs2bE, dtype=torch.long),  # obs2 - observer 1, birdseye graph, E=2
    torch.tensor([4] * obs3bE, dtype=torch.long),  # obs3 - observer 0, birdseye graph, E=6
    torch.tensor([5] * obs3bE, dtype=torch.long),  # obs3 - observer 1, birdseye graph, E=6
    torch.tensor([6] * obs3bE, dtype=torch.long),  # obs3 - observer 2, birdseye graph, E=6
))
glgi_mcn_expected4 = torch.cat((
    torch.tensor([0] * obs1cN ** 2, dtype=torch.long),  # obs1 - observer 0, central graph (node 0), N=2
    torch.tensor([1] * obs1cN ** 2, dtype=torch.long),  # obs1 - observer 1, central graph (node 0), N=2
    torch.tensor([2] * obs2cN ** 2, dtype=torch.long),  # obs2 - observer 0, central graph (node 0), N=3
    torch.tensor([3] * obs2cN ** 2, dtype=torch.long),  # obs2 - observer 1, central graph (node 0), N=3
    torch.tensor([4] * obs3cN ** 2, dtype=torch.long),  # obs3 - observer 0, central graph (node 0), N=3
    torch.tensor([5] * obs3cN ** 2, dtype=torch.long),  # obs3 - observer 1, central graph (node 0), N=3
    torch.tensor([6] * obs3cN ** 2, dtype=torch.long),  # obs3 - observer 2, central graph (node 0), N=3
))
glgi_mcl_expected4 = torch.cat((
    torch.tensor([0] * obs1cE, dtype=torch.long),  # obs1 - observer 0, central graph (node 0), E=2
    torch.tensor([1] * obs1cE, dtype=torch.long),  # obs1 - observer 1, central graph (node 0), E=2
    torch.tensor([2] * obs2cE, dtype=torch.long),  # obs2 - observer 0, central graph (node 0), E=6
    torch.tensor([3] * obs2cE, dtype=torch.long),  # obs2 - observer 1, central graph (node 0), E=6
    torch.tensor([4] * obs3cE, dtype=torch.long),  # obs3 - observer 0, central graph (node 0), E=6
    torch.tensor([5] * obs3cE, dtype=torch.long),  # obs3 - observer 1, central graph (node 0), E=6
    torch.tensor([6] * obs3cE, dtype=torch.long),  # obs3 - observer 2, central graph (node 0), E=6
))
glgi_mln_expected4 = torch.cat((
    torch.tensor([0] * obs1n0N ** 2, dtype=torch.long),  # obs1 - observer 0's graph, N=2
    torch.tensor([1] * obs1n1N ** 2, dtype=torch.long),  # obs1 - observer 1's graph, N=2
    torch.tensor([2] * obs2n0N ** 2, dtype=torch.long),  # obs2 - observer 0's graph, N=3
    torch.tensor([3] * obs2n1N ** 2, dtype=torch.long),  # obs2 - observer 1's graph, N=2
    torch.tensor([4] * obs3n0N ** 2, dtype=torch.long),  # obs3 - observer 0's graph, N=3
    torch.tensor([5] * obs3n1N ** 2, dtype=torch.long),  # obs3 - observer 1's graph, N=3
    torch.tensor([6] * obs3n2N ** 2, dtype=torch.long),  # obs3 - observer 2's graph, N=3
))
glgi_mll_expected4 = torch.cat((
    torch.tensor([0] * obs1n0E, dtype=torch.long),  # obs1 - observer 0's graph, E=2
    torch.tensor([1] * obs1n1E, dtype=torch.long),  # obs1 - observer 1's graph, E=2
    torch.tensor([2] * obs2n0E, dtype=torch.long),  # obs2 - observer 0's graph, E=6
    torch.tensor([3] * obs2n1E, dtype=torch.long),  # obs2 - observer 1's graph, E=2
    torch.tensor([4] * obs3n0E, dtype=torch.long),  # obs3 - observer 0's graph, E=6
    torch.tensor([5] * obs3n1E, dtype=torch.long),  # obs3 - observer 1's graph, E=6
    torch.tensor([6] * obs3n2E, dtype=torch.long),  # obs3 - observer 2's graph, E=6
))

get_logprob_graph_idx_expected4 = [
    glgi_sbn_expected4,
    glgi_sbl_expected4,
    glgi_scn_expected4,
    glgi_scl_expected4,
    glgi_mbn_expected4,
    glgi_mbl_expected4,
    glgi_mcn_expected4,
    glgi_mcl_expected4,
    glgi_mln_expected4,
    glgi_mll_expected4,
]

get_logprob_graph_idx_settings_batch4 = ([("_get_logprob_graph_idx", obs_batch4, expected, func_args) for (expected, func_args)
                                            in zip(get_logprob_graph_idx_expected4, func_arg_list_3args)])

get_logprob_graph_idx_settings = (get_logprob_graph_idx_settings_batch1
                                  + get_logprob_graph_idx_settings_batch2
                                  + get_logprob_graph_idx_settings_batch3
                                  + get_logprob_graph_idx_settings_batch4)

# === get_entropy_mask =======================================================

# actor_loc:
# - single: obtain full mask for the one input graph (birdseye or central)
# - multi: obtain mask for each local input graph. For each graph,
#          the values are true wherever the agent ID matches the observer ID.
# actor_input:
# - birdseye: obtain full mask for the one input graph (birdseye)
# - central: obtain full mask for the one input graph (central)
# algo_config:
# - next-hop: E*N entries per obs graph in obs
# - link-weight: E entries per obs graph in obs

# expected outputs for get_entropy_mask() on obs1
# Obs 1: 6 nodes, 6 edges, observer IDs [-1, 0, 1], agent IDs [0, 1, 0, 1, 0, 1]
gem_sbnn_expected1 = torch.ones(obs1bN ** 2, dtype=torch.bool)
gem_sbnt_expected1 = torch.ones(obs1bE * obs1bN, dtype=torch.bool)
gem_sbl_expected1 = torch.ones(obs1bE, dtype=torch.bool)
gem_scnn_expected1 = torch.ones(obs1cN ** 2, dtype=torch.bool)
gem_scnt_expected1 = torch.ones(obs1cE * obs1cN, dtype=torch.bool)
gem_scl_expected1 = torch.ones(obs1cE, dtype=torch.bool)
gem_mbnn_expected1 = torch.cat((
    torch.tensor([1, 1, 0, 0], dtype=torch.bool),  # observer 0, birdseye graph
    torch.tensor([0, 0, 1, 1], dtype=torch.bool),  # observer 1, birdseye graph
))
gem_mbnt_expected1 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # observer 0, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # observer 1, birdseye graph, N=2, edge_index[0] = [0, 1]
))
gem_mbl_expected1 = torch.cat((
    torch.tensor([1, 0], dtype=torch.bool),  # observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # observer 1, birdseye graph, edge_index[0] = [0, 1]
))
gem_mcnn_expected1 = gem_mbnn_expected1
gem_mcnt_expected1 = gem_mbnt_expected1  # birdseye and central graphs are the same in obs1
gem_mcl_expected1 = gem_mbl_expected1  # birdseye and central graphs are the same in obs1
gem_mlnn_expected1 = gem_mbnn_expected1  # local and birdseye graphs are the same in obs1
gem_mlnt_expected1 = gem_mbnt_expected1  # local and birdseye graphs are the same in obs1
gem_mll_expected1 = gem_mbl_expected1  # local and birdseye graphs are the same in obs1

get_entropy_mask_expected1 = [
    gem_sbnn_expected1,
    gem_sbnt_expected1,
    gem_sbl_expected1,
    gem_scnn_expected1,
    gem_scnt_expected1,
    gem_scl_expected1,
    gem_mbnn_expected1,
    gem_mbnt_expected1,
    gem_mbl_expected1,
    gem_mcnn_expected1,
    gem_mcnt_expected1,
    gem_mcl_expected1,
    gem_mlnn_expected1,
    gem_mlnt_expected1,
    gem_mll_expected1
]

get_entropy_mask_settings_obs1 = ([("_get_entropy_mask", obs1, expected, func_args) for (expected, func_args)
                                   in zip(get_entropy_mask_expected1, func_arg_list_4args)])

# expected outputs for get_entropy_mask() on obs2
# Obs 2: 7 nodes, 10 edges, observer IDs [-1, 0, 1], agent IDs [0, 1, 0, 1, 2, 0, 1]
gem_sbnn_expected2 = torch.ones(obs2bN ** 2, dtype=torch.bool)
gem_sbnt_expected2 = torch.ones(obs2bE * obs2bN, dtype=torch.bool)
gem_sbl_expected2 = torch.ones(obs2bE, dtype=torch.bool)
gem_scnn_expected2 = torch.ones(obs2cN ** 2, dtype=torch.bool)
gem_scnt_expected2 = torch.ones(obs2cE * obs2cN, dtype=torch.bool)
gem_scl_expected2 = torch.ones(obs2cE, dtype=torch.bool)
gem_mbnn_expected2 = torch.cat((  # both observer 0 and 1 get birdseye graph (observer -1's graph w/ 2 nodes)
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # observer 0, birdseye graph
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # observer 1, birdseye graph
))
gem_mbnt_expected2 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # observer 0, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # observer 1, birdseye graph, N=2, edge_index[0] = [0, 1]
))
gem_mbl_expected2 = torch.cat((
    torch.tensor([1, 0], dtype=torch.bool),  # observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # observer 1, birdseye graph, edge_index[0] = [0, 1]
))
gem_mcnn_expected2 = torch.cat((  # both observer 0 and 1 get central graph (observer 0's graph w/ 3 nodes)
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # observer 0, central graph
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # observer 0, central graph
))
gem_mcnt_expected2 = torch.cat((  # both observer 0 and 1 get central graph (observer 0's graph)
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
))
gem_mcl_expected2 = torch.cat((  # both observer 0 and 1 get central graph (observer 0's graph)
    torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),  # N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.bool),  # N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
))
gem_mlnn_expected2 = torch.cat((
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # observer 0's graph, N=3
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # observer 1's graph, N=2
))
gem_mlnt_expected2 = torch.cat((  # both observer 0 and 1 get their own respective graphs
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # observer 0's graph, N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 1, 1], dtype=torch.bool),  # observer 1's graph, N=2, edge_index[0] = [0, 1]
))
gem_mll_expected2 = torch.cat((
    torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),  # observer 0's graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 1], dtype=torch.bool),  # observer 1's graph, edge_index[0] = [0, 1]
))

get_entropy_mask_expected2 = [
    gem_sbnn_expected2,
    gem_sbnt_expected2,
    gem_sbl_expected2,
    gem_scnn_expected2,
    gem_scnt_expected2,
    gem_scl_expected2,
    gem_mbnn_expected2,
    gem_mbnt_expected2,
    gem_mbl_expected2,
    gem_mcnn_expected2,
    gem_mcnt_expected2,
    gem_mcl_expected2,
    gem_mlnn_expected2,
    gem_mlnt_expected2,
    gem_mll_expected2
]

get_entropy_mask_settings_obs2 = ([("_get_entropy_mask", obs2, expected, func_args) for (expected, func_args)
                                   in zip(get_entropy_mask_expected2, func_arg_list_4args)])

# expected outputs for get_entropy_mask() on obs3
# Obs 3: 12 nodes, 24 edges, observer IDs [-1, 0, 1, 2], agent IDs [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
gem_sbnn_expected3 = torch.ones(obs3bN ** 2, dtype=torch.bool)
gem_sbnt_expected3 = torch.ones(obs3bE * obs3bN, dtype=torch.bool)
gem_sbl_expected3 = torch.ones(obs3bE, dtype=torch.bool)
gem_scnn_expected3 = torch.ones(obs3cN ** 2, dtype=torch.bool)
gem_scnt_expected3 = torch.ones(obs3cE * obs3cN, dtype=torch.bool)
gem_scl_expected3 = torch.ones(obs3cE, dtype=torch.bool)
gem_mbnn_expected3 = torch.cat((
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # observer 0, birdseye graph
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                    0, 0, 0], dtype=torch.bool),  # observer 1, birdseye graph
    torch.tensor([0, 0, 0,
                  0, 0, 0,
                    1, 1, 1], dtype=torch.bool),  # observer 2, birdseye graph
))
gem_mbnt_expected3 = torch.cat((  # observer 0, 1, and 2 get birdseye graph (observer -1's graph)
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0,
                  0, 0, 0,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  1, 1, 1], dtype=torch.bool),  # N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
))
gem_mbl_expected3 = torch.cat((
    torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),  # N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.bool),  # N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0, 1, 0, 1], dtype=torch.bool),  # N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
))
gem_mcnn_expected3 = gem_mbnn_expected3
gem_mcnt_expected3 = gem_mbnt_expected3  # birdseye and central graphs are the same in obs3
gem_mcl_expected3 = gem_mbl_expected3  # birdseye and central graphs are the same in obs3
gem_mlnn_expected3 = gem_mbnn_expected3
gem_mlnt_expected3 = gem_mbnt_expected3  # local and birdseye graphs are the same in obs3
gem_mll_expected3 = gem_mbl_expected3  # local and birdseye graphs are the same in obs3

get_entropy_mask_expected3 = [
    gem_sbnn_expected3,
    gem_sbnt_expected3,
    gem_sbl_expected3,
    gem_scnn_expected3,
    gem_scnt_expected3,
    gem_scl_expected3,
    gem_mbnn_expected3,
    gem_mbnt_expected3,
    gem_mbl_expected3,
    gem_mcnn_expected3,
    gem_mcnt_expected3,
    gem_mcl_expected3,
    gem_mlnn_expected3,
    gem_mlnt_expected3,
    gem_mll_expected3
]

get_entropy_mask_settings_obs3 = ([("_get_entropy_mask", obs3, expected, func_args) for (expected, func_args)
                                   in zip(get_entropy_mask_expected3, func_arg_list_4args)])

# expected outputs for get_entropy_mask() on batch1 = [obs1, obs2]
gem_sbnn_expected_batch1 = torch.ones(obs1bN ** 2 + obs2bN ** 2, dtype=torch.bool)
gem_sbnt_expected_batch1 = torch.ones(obs1bE * obs1bN + obs2bE * obs2bN, dtype=torch.bool)
gem_sbl_expected_batch1 = torch.ones(obs1bE + obs2bE, dtype=torch.bool)
gem_scnn_expected_batch1 = torch.ones(obs1cN ** 2 + obs2cN ** 2, dtype=torch.bool)
gem_scnt_expected_batch1 = torch.ones(obs1cE * obs1cN + obs2cE * obs2cN, dtype=torch.bool)
gem_scl_expected_batch1 = torch.ones(obs1cE + obs2cE, dtype=torch.bool)
gem_mbnn_expected_batch1 = torch.cat((gem_mbnn_expected1, gem_mbnn_expected2))
gem_mbnt_expected_batch1 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1, observer 0, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1, observer 1, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs2, observer 0, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2, observer 1, birdseye graph, N=2, edge_index[0] = [0, 1]
))
gem_mbl_expected_batch1 = torch.cat((
    torch.tensor([1, 0], dtype=torch.bool),  # obs1, observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs1, observer 1, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 0], dtype=torch.bool),  # obs2, observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs2, observer 1, birdseye graph, edge_index[0] = [0, 1]
))
gem_mcnn_expected_batch1 = torch.cat((gem_mcnn_expected1, gem_mcnn_expected2))
gem_mcnt_expected_batch1 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1, observer 0, central graph (node 0), N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1, observer 1, central graph (node 0), N=2, edge_index[0] = [0, 1]
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2, observer 0, central graph (node 0), N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # obs2, observer 1, central graph (node 0), N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
))
gem_mcl_expected_batch1 = torch.cat((
    torch.tensor([1, 0], dtype=torch.bool),  # obs1, observer 0, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs1, observer 1, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),  # obs2, observer 0, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.bool),  # obs2, observer 1, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
))
gem_mlnn_expected_batch1 = torch.cat((gem_mlnn_expected1, gem_mlnn_expected2))
gem_mlnt_expected_batch1 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1, observer 0's graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1, observer 1's graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2, observer 0's graph, N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2, observer 1's graph, N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
))
gem_mll_expected_batch1 = torch.cat((
    torch.tensor([1, 0], dtype=torch.bool),  # obs1, observer 0's graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs1, observer 1's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),  # obs2, observer 0's graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 1], dtype=torch.bool),  # obs2, observer 1's graph, edge_index[0] = [0, 1]
))

get_entropy_mask_expected_batch1 = [
    gem_sbnn_expected_batch1,
    gem_sbnt_expected_batch1,
    gem_sbl_expected_batch1,
    gem_scnn_expected_batch1,
    gem_scnt_expected_batch1,
    gem_scl_expected_batch1,
    gem_mbnn_expected_batch1,
    gem_mbnt_expected_batch1,
    gem_mbl_expected_batch1,
    gem_mcnn_expected_batch1,
    gem_mcnt_expected_batch1,
    gem_mcl_expected_batch1,
    gem_mlnn_expected_batch1,
    gem_mlnt_expected_batch1,
    gem_mll_expected_batch1
]

get_entropy_mask_settings_batch1 = ([("_get_entropy_mask", obs_batch1, expected, func_args)
                                     for (expected, func_args)
                                     in zip(get_entropy_mask_expected_batch1, func_arg_list_4args)])

# expected outputs for get_entropy_mask() on batch2 = [obs2, obs1]
gem_sbnn_expected_batch2 = torch.ones(obs2bN ** 2 + obs1bN ** 2, dtype=torch.bool)
gem_sbnt_expected_batch2 = torch.ones(obs2bE * obs2bN + obs1bE * obs1bN, dtype=torch.bool)
gem_sbl_expected_batch2 = torch.ones(obs2bE + obs1bE, dtype=torch.bool)
gem_scnn_expected_batch2 = torch.ones(obs2cN ** 2 + obs1cN ** 2, dtype=torch.bool)
gem_scnt_expected_batch2 = torch.ones(obs2cE * obs2cN + obs1cE * obs1cN, dtype=torch.bool)
gem_scl_expected_batch2 = torch.ones(obs2cE + obs1cE, dtype=torch.bool)
gem_mbnn_expected_batch2 = torch.cat((gem_mbnn_expected2, gem_mbnn_expected1))
gem_mbnt_expected_batch2 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs2, observer 0, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2, observer 1, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1, observer 0, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1, observer 1, birdseye graph, N=2, edge_index[0] = [0, 1]
))
gem_mbl_expected_batch2 = torch.cat((
    torch.tensor([1, 0], dtype=torch.bool),  # obs2, observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs2, observer 1, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 0], dtype=torch.bool),  # obs1, observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs1, observer 1, birdseye graph, edge_index[0] = [0, 1]
))
gem_mcnn_expected_batch2 = torch.cat((gem_mcnn_expected2, gem_mcnn_expected1))
gem_mcnt_expected_batch2 = torch.cat((
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2, observer 0, central graph (node 0), N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # obs2, observer 1, central graph (node 0), N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1, observer 0, central graph (node 0), N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1, observer 1, central graph (node 0), N=2, edge_index[0] = [0, 1]
))
gem_mcl_expected_batch2 = torch.cat((
    torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),  # obs2, observer 0, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.bool),  # obs2, observer 1, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 0], dtype=torch.bool),  # obs1, observer 0, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs1, observer 1, central graph (node 0), edge_index[0] = [0, 1]
))
gem_mlnn_expected_batch2 = torch.cat((gem_mlnn_expected2, gem_mlnn_expected1))
gem_mlnt_expected_batch2 = torch.cat((
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2, observer 0's graph, N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2, observer 1's graph, N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1, observer 0's graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1, observer 1's graph, N=2, edge_index[0] = [0, 1]
))
gem_mll_expected_batch2 = torch.cat((
    torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),  # obs2, observer 0's graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 1], dtype=torch.bool),  # obs2, observer 1's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 0], dtype=torch.bool),  # obs1, observer 0's graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs1, observer 1's graph, edge_index[0] = [0, 1]
))

get_entropy_mask_expected_batch2 = [
    gem_sbnn_expected_batch2,
    gem_sbnt_expected_batch2,
    gem_sbl_expected_batch2,
    gem_scnn_expected_batch2,
    gem_scnt_expected_batch2,
    gem_scl_expected_batch2,
    gem_mbnn_expected_batch2,
    gem_mbnt_expected_batch2,
    gem_mbl_expected_batch2,
    gem_mcnn_expected_batch2,
    gem_mcnt_expected_batch2,
    gem_mcl_expected_batch2,
    gem_mlnn_expected_batch2,
    gem_mlnt_expected_batch2,
    gem_mll_expected_batch2
]

get_entropy_mask_settings_batch2 = ([("_get_entropy_mask", obs_batch2, expected, func_args)
                                     for (expected, func_args)
                                     in zip(get_entropy_mask_expected_batch2, func_arg_list_4args)])

# expected outputs for get_entropy_mask() on batch3 = [obs1, obs2, obs1]
gem_sbnn_expected_batch3 = torch.ones(obs1bN ** 2 + obs2bN ** 2 + obs1bN ** 2, dtype=torch.bool)
gem_sbnt_expected_batch3 = torch.ones(obs1bE * obs1bN + obs2bE * obs2bN + obs1bE * obs1bN, dtype=torch.bool)
gem_sbl_expected_batch3 = torch.ones(obs1bE + obs2bE + obs1bE, dtype=torch.bool)
gem_scnn_expected_batch3 = torch.ones(obs1cN ** 2 + obs2cN ** 2 + obs1cN ** 2, dtype=torch.bool)
gem_scnt_expected_batch3 = torch.ones(obs1cE * obs1cN + obs2cE * obs2cN + obs1cE * obs1cN, dtype=torch.bool)
gem_scl_expected_batch3 = torch.ones(obs1cE + obs2cE + obs1cE, dtype=torch.bool)
gem_mbnn_expected_batch3 = torch.cat((gem_mbnn_expected1, gem_mbnn_expected2, gem_mbnn_expected1))
gem_mbnt_expected_batch3 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1, observer 0, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1, observer 1, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs2, observer 0, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2, observer 1, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1, observer 0, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1, observer 1, birdseye graph, N=2, edge_index[0] = [0, 1]
))
gem_mbl_expected_batch3 = torch.cat((
    torch.tensor([1, 0], dtype=torch.bool),  # obs1, observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs1, observer 1, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 0], dtype=torch.bool),  # obs2, observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs2, observer 1, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 0], dtype=torch.bool),  # obs1, observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs1, observer 1, birdseye graph, edge_index[0] = [0, 1]
))
gem_mcnn_expected_batch3 = torch.cat((gem_mcnn_expected1, gem_mcnn_expected2, gem_mcnn_expected1))
gem_mcnt_expected_batch3 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1, observer 0, central graph (node 0), N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1, observer 1, central graph (node 0), N=2, edge_index[0] = [0, 1]
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2, observer 0, central graph (node 0), N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # obs2, observer 1, central graph (node 0), N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1, observer 0, central graph (node 0), N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1, observer 1, central graph (node 0), N=2, edge_index[0] = [0, 1]
))
gem_mcl_expected_batch3 = torch.cat((
    torch.tensor([1, 0], dtype=torch.bool),  # obs1, observer 0, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs1, observer 1, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),  # obs2, observer 0, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.bool),  # obs2, observer 1, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 0], dtype=torch.bool),  # obs1, observer 0, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs1, observer 1, central graph (node 0), edge_index[0] = [0, 1]
))
gem_mlnn_expected_batch3 = torch.cat((gem_mlnn_expected1, gem_mlnn_expected2, gem_mlnn_expected1))
gem_mlnt_expected_batch3 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1, observer 0's graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1, observer 1's graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2, observer 0's graph, N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2, observer 1's graph, N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1, observer 0's graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1, observer 1's graph, N=2, edge_index[0] = [0, 1]
))
gem_mll_expected_batch3 = torch.cat((
    torch.tensor([1, 0], dtype=torch.bool),  # obs1, observer 0's graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs1, observer 1's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),  # obs2, observer 0's graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 1], dtype=torch.bool),  # obs2, observer 1's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 0], dtype=torch.bool),  # obs1, observer 0's graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs1, observer 1's graph, edge_index[0] = [0, 1]
))

get_entropy_mask_expected_batch3 = [
    gem_sbnn_expected_batch3,
    gem_sbnt_expected_batch3,
    gem_sbl_expected_batch3,
    gem_scnn_expected_batch3,
    gem_scnt_expected_batch3,
    gem_scl_expected_batch3,
    gem_mbnn_expected_batch3,
    gem_mbnt_expected_batch3,
    gem_mbl_expected_batch3,
    gem_mcnn_expected_batch3,
    gem_mcnt_expected_batch3,
    gem_mcl_expected_batch3,
    gem_mlnn_expected_batch3,
    gem_mlnt_expected_batch3,
    gem_mll_expected_batch3
]

get_entropy_mask_settings_batch3 = ([("_get_entropy_mask", obs_batch3, expected, func_args)
                                     for (expected, func_args)
                                     in zip(get_entropy_mask_expected_batch3, func_arg_list_4args)])

# expected outputs for get_entropy_mask() on batch4 = [obs1, obs2, obs3]
gem_sbnn_expected_batch4 = torch.ones(obs1bN ** 2 + obs2bN ** 2 + obs3bN ** 2, dtype=torch.bool)
gem_sbnt_expected_batch4 = torch.ones(obs1bE * obs1bN + obs2bE * obs2bN + obs3bE * obs3bN, dtype=torch.bool)
gem_sbl_expected_batch4 = torch.ones(obs1bE + obs2bE + obs3bE, dtype=torch.bool)
gem_scnn_expected_batch4 = torch.ones(obs1cN ** 2 + obs2cN ** 2 + obs3cN ** 2, dtype=torch.bool)
gem_scnt_expected_batch4 = torch.ones(obs1cE * obs1cN + obs2cE * obs2cN + obs3cE * obs3cN, dtype=torch.bool)
gem_scl_expected_batch4 = torch.ones(obs1cE + obs2cE + obs3cE, dtype=torch.bool)
gem_mbnn_expected_batch4 = torch.cat((gem_mbnn_expected1, gem_mbnn_expected2, gem_mbnn_expected3))
gem_mbnt_expected_batch4 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1, observer 0, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1, observer 1, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs2, observer 0, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2, observer 1, birdseye graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs3, observer 0, birdseye graph, N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # obs3, observer 1, birdseye graph, N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0,
                  0, 0, 0,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  1, 1, 1], dtype=torch.bool),  # obs3, observer 2, birdseye graph, N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
))
gem_mbl_expected_batch4 = torch.cat((
    torch.tensor([1, 0], dtype=torch.bool),  # obs1, observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs1, observer 1, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 0], dtype=torch.bool),  # obs2, observer 0, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs2, observer 1, birdseye graph, edge_index[0] = [0, 1]
    torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),  # obs3, observer 0, birdseye graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.bool),  # obs3, observer 1, birdseye graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0, 1, 0, 1], dtype=torch.bool),  # obs3, observer 2, birdseye graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
))
gem_mcnn_expected_batch4 = torch.cat((gem_mcnn_expected1, gem_mcnn_expected2, gem_mcnn_expected3))
gem_mcnt_expected_batch4 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1, observer 0, central graph (node 0), N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1, observer 1, central graph (node 0), N=2, edge_index[0] = [0, 1]
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2, observer 0, central graph (node 0), N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # obs2, observer 1, central graph (node 0), N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs3, observer 0, central_graph (node 0), N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # obs3, observer 1, central_graph (node 0), N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0,
                  0, 0, 0,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  1, 1, 1], dtype=torch.bool),  # obs3, observer 2, central_graph (node 0), N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
))
gem_mcl_expected_batch4 = torch.cat((
    torch.tensor([1, 0], dtype=torch.bool),  # obs1, observer 0, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs1, observer 1, central graph (node 0), edge_index[0] = [0, 1]
    torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),  # obs2, observer 0, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.bool),  # obs2, observer 1, central graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),  # obs3, observer 0, central_graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.bool),  # obs3, observer 1, central_graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0, 1, 0, 1], dtype=torch.bool),  # obs3, observer 2, central_graph (node 0), edge_index[0] = [0, 1, 0, 2, 1, 2]
))
gem_mlnn_expected_batch4 = torch.cat((gem_mlnn_expected1, gem_mlnn_expected2, gem_mlnn_expected3))
gem_mlnt_expected_batch4 = torch.cat((
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # obs1, observer 0's graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs1, observer 1's graph, N=2, edge_index[0] = [0, 1]
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs2, observer 0's graph, N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # obs2, observer 1's graph, N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # obs3, observer 0's graph, N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # obs3, observer 1's graph, N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0,
                  0, 0, 0,
                  0, 0, 0,
                  1, 1, 1,
                  0, 0, 0,
                  1, 1, 1], dtype=torch.bool),  # obs3, observer 2's graph, N=3, edge_index[0] = [0, 1, 0, 2, 1, 2]
))
gem_mll_expected_batch4 = torch.cat((
    torch.tensor([1, 0], dtype=torch.bool),  # obs1, observer 0's graph, edge_index[0] = [0, 1]
    torch.tensor([0, 1], dtype=torch.bool),  # obs1, observer 1's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),  # obs2, observer 0's graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 1], dtype=torch.bool),  # obs2, observer 1's graph, edge_index[0] = [0, 1]
    torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool),  # obs3, observer 0's graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.bool),  # obs3, observer 1's graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
    torch.tensor([0, 0, 0, 1, 0, 1], dtype=torch.bool),  # obs3, observer 2's graph, edge_index[0] = [0, 1, 0, 2, 1, 2]
))

get_entropy_mask_expected_batch4 = [
    gem_sbnn_expected_batch4,
    gem_sbnt_expected_batch4,
    gem_sbl_expected_batch4,
    gem_scnn_expected_batch4,
    gem_scnt_expected_batch4,
    gem_scl_expected_batch4,
    gem_mbnn_expected_batch4,
    gem_mbnt_expected_batch4,
    gem_mbl_expected_batch4,
    gem_mcnn_expected_batch4,
    gem_mcnt_expected_batch4,
    gem_mcl_expected_batch4,
    gem_mlnn_expected_batch4,
    gem_mlnt_expected_batch4,
    gem_mll_expected_batch4
]

get_entropy_mask_settings_batch4 = ([("_get_entropy_mask", obs_batch4, expected, func_args)
                                     for (expected, func_args)
                                     in zip(get_entropy_mask_expected_batch4, func_arg_list_4args)])

get_entropy_mask_settings = (get_entropy_mask_settings_obs1
                                + get_entropy_mask_settings_obs2
                                + get_entropy_mask_settings_obs3
                                + get_entropy_mask_settings_batch1
                                + get_entropy_mask_settings_batch2
                                + get_entropy_mask_settings_batch3
                                + get_entropy_mask_settings_batch4)

# =========================================================================

all_settings = (get_logprob_mask_settings
                + get_logprob_graph_idx_settings
                + get_entropy_mask_settings
                )


@pytest.fixture(scope="session")
def base_cfg():
    # Make sure Hydra's global state is fresh for each test
    GlobalHydra.instance().clear()
    cfg: DictConfig = None
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="base_config")
    return cfg


@pytest.mark.parametrize("setting", all_settings)
def test_get_mask_or_offset(setting, base_cfg):

    func_name: str
    input_obs: Observation | ObservationBatch
    expected_output: torch.Tensor
    func_args: dict
    func_name, input_obs, expected_output, func_args = setting

    if expected_output is NotImplemented:
        pytest.skip("This test is not implemented yet.")

    mock_env = None  # mock env is not needed for these tests

    algo_config = base_cfg.algorithm.trainable.copy()
    algo_config.flavor = func_args["algo_config"]
    algo = get_trainable_algorithm(mock_env, algo_config)

    actor_loc = func_args["actor_loc"]
    actor_input = func_args["actor_input"]
    algo.set_loc(actor_loc, actor_input)

    trainer_config = base_cfg.trainer.ppo.copy()
    trainer_config.flavor = "PPO" if actor_loc == "single" else "MAPPO"
    trainer_config.module_input = actor_input
    if "ppo_fl_style" in func_args:
        trainer_config.fieldlines.style = func_args["ppo_fl_style"]
    ppo = get_ppo_trainer(base_cfg, trainer_config, mock_env, algo)

    input_obs_view = algo.get_actor_view(input_obs)
    result = getattr(ppo, func_name)(input_obs_view)

    assert torch.equal(result, expected_output), f"Expected {expected_output} but got {result}"
