import sys
sys.path.append(".")

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig
import pytest

from ._mock_obs import *
from src.algorithm.trainable import get_trainable_algorithm


# === func args =============================================================

sb_func_args = {"actor_loc": "single", "actor_input": "birdseye"}
sc_func_args = {"actor_loc": "single", "actor_input": "central"}
mb_func_args = {"actor_loc": "multi", "actor_input": "birdseye"}
mc_func_args = {"actor_loc": "multi", "actor_input": "central"}
ml_func_args = {"actor_loc": "multi", "actor_input": "local"}

func_arg_list_2args = [sb_func_args, sc_func_args, mb_func_args, mc_func_args, ml_func_args]

# === get_action_sedi_offsets (2arg) =======================================================

# expected outputs for get_action_sedi_offsets() on obs1
gaso_sb_expected1 = torch.zeros(obs1bN ** 2, dtype=torch.long)
gaso_sc_expected1 = torch.zeros(obs1cN ** 2, dtype=torch.long)
gaso_mb_expected1 = torch.cat((
    torch.zeros(obs1bN ** 2, dtype=torch.long),  # observer 0, birdseye graph
    torch.ones(obs1bN ** 2, dtype=torch.long) * obs1bN * obs1bE,  # observer 1, birdseye graph
))
gaso_mc_expected1 = torch.cat((
    torch.zeros(obs1cN ** 2, dtype=torch.long),  # observer 0, central graph
    torch.ones(obs1cN ** 2, dtype=torch.long) * obs1cN * obs1cE,  # observer 1, central graph
))
gaso_ml_expected1 = torch.cat((
    torch.zeros(obs1n0N ** 2, dtype=torch.long),  # observer 0's graph
    torch.ones(obs1n0N ** 2, dtype=torch.long) * obs1n0N * obs1n0E,  # observer 1's graph
))

get_action_sedi_offsets_expected1 = [
    gaso_sb_expected1,
    gaso_sc_expected1,
    gaso_mb_expected1,
    gaso_mc_expected1,
    gaso_ml_expected1,
]

get_action_sedi_offsets_settings_obs1 = ([("get_action_sedi_offsets", obs1, expected, func_args) for (expected, func_args)
                                          in zip(get_action_sedi_offsets_expected1, func_arg_list_2args)])

gaso_sb_expected2 = torch.zeros(obs2bN ** 2, dtype=torch.long)
gaso_sc_expected2 = torch.zeros(obs2cN ** 2, dtype=torch.long)
gaso_mb_expected2 = torch.cat((
    torch.zeros(obs2bN ** 2, dtype=torch.long),  # observer 0, birdseye graph
    torch.ones(obs2bN ** 2, dtype=torch.long) * obs2bN * obs2bE,  # observer 1, birdseye graph
))
gaso_mc_expected2 = torch.cat((
    torch.zeros(obs2cN ** 2, dtype=torch.long),  # observer 0, central graph
    torch.ones(obs2cN ** 2, dtype=torch.long) * obs2cN * obs2cE,  # observer 1, central graph
))
gaso_ml_expected2 = torch.cat((
    torch.zeros(obs2n0N ** 2, dtype=torch.long),  # observer 0's graph
    torch.ones(obs2n1N ** 2, dtype=torch.long) * obs2n0N * obs2n0E,  # observer 1's graph
))

# expected outputs for get_action_sedi_offsets() on obs2
get_action_sedi_offsets_expected2 = [
    gaso_sb_expected2,
    gaso_sc_expected2,
    gaso_mb_expected2,
    gaso_mc_expected2,
    gaso_ml_expected2,
]

get_action_sedi_offsets_settings_obs2 = ([("get_action_sedi_offsets", obs2, expected, func_args) for (expected, func_args)
                                          in zip(get_action_sedi_offsets_expected2, func_arg_list_2args)])

# expected outputs for get_action_sedi_offsets() on obs3
gaso_sb_expected3 = torch.zeros(obs3bN ** 2, dtype=torch.long)
gaso_sc_expected3 = torch.zeros(obs3cN ** 2, dtype=torch.long)
gaso_mb_expected3 = torch.cat((
    torch.zeros(obs3bN ** 2, dtype=torch.long),  # observer 0, birdseye graph
    torch.ones(obs3bN ** 2, dtype=torch.long) * obs3bN * obs3bE,  # observer 1, birdseye graph
    torch.ones(obs3bN ** 2, dtype=torch.long) * obs3bN * obs3bE * 2,  # observer 2, birdseye graph
))
gaso_mc_expected3 = gaso_mb_expected3  # birdseye and central graphs are the same in obs3
gaso_ml_expected3 = gaso_mb_expected3  # local and birdseye graphs are the same in obs3

get_action_sedi_offsets_expected3 = [
    gaso_sb_expected3,
    gaso_sc_expected3,
    gaso_mb_expected3,
    gaso_mc_expected3,
    gaso_ml_expected3,
]

get_action_sedi_offsets_settings_obs3 = ([("get_action_sedi_offsets", obs3, expected, func_args) for (expected, func_args)
                                          in zip(get_action_sedi_offsets_expected3, func_arg_list_2args)])


get_action_sedi_offsets_settings = (get_action_sedi_offsets_settings_obs1
                                    + get_action_sedi_offsets_settings_obs2
                                    + get_action_sedi_offsets_settings_obs3)

# === get_action_sedi_mask (2arg) =======================================================

# expected outputs for get_action_sedi_mask() on obs1
# observer_ids: [-1, 0, 1], agent_ids: [0, 1, 0, 1, 0, 1]
gasm_sb_expected1 = torch.ones(obs1bN ** 2, dtype=torch.bool)
gasm_sc_expected1 = torch.ones(obs1cN ** 2, dtype=torch.bool)
gasm_mb_expected1 = torch.cat((
    torch.tensor([1, 1, 0, 0], dtype=torch.bool),  # observer 0, birdseye graph
    torch.tensor([0, 0, 1, 1], dtype=torch.bool),  # observer 1, birdseye graph
))
gasm_mc_expected1 = gasm_mb_expected1  # birdseye and central graphs are the same in obs1
gasm_ml_expected1 = gasm_mb_expected1  # local and birdseye graphs are the same in obs1

get_action_sedi_mask_expected1 = [
    gasm_sb_expected1,
    gasm_sc_expected1,
    gasm_mb_expected1,
    gasm_mc_expected1,
    gasm_ml_expected1,
]

get_action_sedi_mask_settings_obs1 = ([("get_action_sedi_mask", obs1, expected, func_args) for (expected, func_args)
                                        in zip(get_action_sedi_mask_expected1, func_arg_list_2args)])

# expected outputs for get_action_sedi_mask() on obs2
gasm_sb_expected2 = torch.ones(obs2bN ** 2, dtype=torch.bool)
gasm_sc_expected2 = torch.ones(obs2cN ** 2, dtype=torch.bool)
gasm_mb_expected2 = torch.cat((  # both observer 0 and 1 get birdseye graph (observer -1's graph w/ 2 nodes)
    torch.tensor([1, 1,
                  0, 0], dtype=torch.bool),  # observer 0, birdseye graph
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # observer 1, birdseye graph
))
gasm_mc_expected2 = torch.cat((  # both observer 0 and 1 get central graph (observer 0's graph w/ 3 nodes)
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # observer 0, central graph
    torch.tensor([0, 0, 0,
                  1, 1, 1,
                  0, 0, 0], dtype=torch.bool),  # observer 0, central graph
))
gasm_ml_expected2 = torch.cat((
    torch.tensor([1, 1, 1,
                  0, 0, 0,
                  0, 0, 0], dtype=torch.bool),  # observer 0's graph, N=3
    torch.tensor([0, 0,
                  1, 1], dtype=torch.bool),  # observer 1's graph, N=2
))

get_action_sedi_mask_expected2 = [
    gasm_sb_expected2,
    gasm_sc_expected2,
    gasm_mb_expected2,
    gasm_mc_expected2,
    gasm_ml_expected2,
]

get_action_sedi_mask_settings_obs2 = ([("get_action_sedi_mask", obs2, expected, func_args) for (expected, func_args)
                                        in zip(get_action_sedi_mask_expected2, func_arg_list_2args)])

# expected outputs for get_action_sedi_mask() on obs3
gasm_sb_expected3 = torch.ones(obs3bN ** 2, dtype=torch.bool)
gasm_sc_expected3 = torch.ones(obs3cN ** 2, dtype=torch.bool)
gasm_mb_expected3 = torch.cat((
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
gasm_mc_expected3 = gasm_mb_expected3  # birdseye and central graphs are the same in obs3
gasm_ml_expected3 = gasm_mb_expected3  # local and birdseye graphs are the same in obs3

get_action_sedi_mask_expected3 = [
    gasm_sb_expected3,
    gasm_sc_expected3,
    gasm_mb_expected3,
    gasm_mc_expected3,
    gasm_ml_expected3,
]

get_action_sedi_mask_settings_obs3 = ([("get_action_sedi_mask", obs3, expected, func_args) for (expected, func_args)
                                       in zip(get_action_sedi_mask_expected3, func_arg_list_2args)])

get_action_sedi_mask_settings = (get_action_sedi_mask_settings_obs1
                                 + get_action_sedi_mask_settings_obs2
                                 + get_action_sedi_mask_settings_obs3
                                 )

# =========================================================================

all_settings = get_action_sedi_offsets_settings + get_action_sedi_mask_settings


# trianable algo config for all masking tests (the masking functions do not depend on the config)
TRAINABLE_CONFIG = OmegaConf.create({
    "fieldlines": OmegaConf.load("config/algorithm/trainable/fieldlines/default_fl.yaml"),
    "feat": OmegaConf.load("config/algorithm/trainable/feat/default_feat.yaml"),
    "nn": OmegaConf.load("config/algorithm/trainable/nn/default_nn.yaml"),
    "flavor": "fieldlines",
    "load_dir": "",
    "normalize_observations": True,
})


@pytest.fixture(scope="session")
def base_cfg():
    # Make sure Hydra's global state is fresh for each test
    GlobalHydra.instance().clear()
    cfg: DictConfig = None
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="base_config")
    return cfg


@pytest.mark.parametrize("setting", all_settings)
def test_trainable_algo_masks(setting, base_cfg):

    func_name: str
    input_obs: Observation
    expected_output: torch.Tensor
    func_args: dict
    func_name, input_obs, expected_output, func_args = setting

    if expected_output is NotImplemented:
        pytest.skip("This test is not implemented yet.")

    mock_env = None  # mock env is not needed for these tests
    trainable_config = base_cfg.algorithm.trainable
    algo = get_trainable_algorithm(mock_env, trainable_config)

    actor_loc = func_args["actor_loc"]
    actor_input = func_args["actor_input"]
    algo.set_loc(actor_loc, actor_input)

    input_obs_view = algo.get_actor_view(input_obs)
    result = getattr(algo, func_name)(input_obs_view)

    assert torch.equal(result, expected_output), f"Expected {expected_output} but got {result}"
