from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn

from src.algorithm.algorithm import Algorithm
from src.normalizer.obs import get_obs_normalizer, ObservationNormalizer
from src.scenario import NetworkScenario
from src.utils.features import (get_acceptable_features,
                                global_feat_in_struct_order,
                                node_feat_in_struct_order,
                                edge_feat_in_struct_order)
from src.utils.logging import LOG_CONSOLE_RUN, LOG_CONSOLE_INFO
from src.utils.observations import Observation
from src.utils.canonical_bytes import canonical_hash
from src.utils.types import ActionRaw, ActionSedi, Reward, LogprobTensor, ValueTensor


class TrainableAlgorithm(Algorithm):
    """
    Abstract class for trainable algorithms with PyTorch models.
    """
    name: str  # name of the algorithm

    def __init__(self,
                 env,
                 trainable_config: DictConfig,
                 save_path: Path | None,
                 load_path: Path | None,
                 device: str = "cpu"):
        super().__init__(env)

        # obtain acceptable/input features, filtering indices and feature counts as neural network input dimensions
        self._prepare_features(trainable_config.feat)

        # initialize the actor module
        self.actor: nn.Module = self._get_actor(trainable_config.nn, device)

        # load actor and obs. normalizer if specified, or initialize normalizer
        if load_path is not None:
            self.load(load_path)
        else:
            self.obs_normalizer = get_obs_normalizer(trainable_config.normalize_observations,
                                                     self.feature_counts)

        # save trainable config (required to properly load the model later)
        self.save_path = save_path
        if self.save_path is None:
            LOG_CONSOLE_INFO("No save path specified, model will not be saved to disk.")
        else:
            with open(self.save_path / "trainable_config.yaml", "w") as f:
                OmegaConf.save(trainable_config, f)

        # module will remain in eval mode except for when updating parameters
        self.set_train(False)

    def _prepare_features(self, feat_config: DictConfig):
        self.acceptable_features = get_acceptable_features(feat_config)
        self.acceptable_feat_idx = {
            "global": [global_feat_in_struct_order.index(feat) for feat in self.acceptable_features['global']],
            "node": [node_feat_in_struct_order.index(feat) for feat in self.acceptable_features['node']
                     if feat != "spdist"],
            "edge": [edge_feat_in_struct_order.index(feat) for feat in self.acceptable_features['edge']
                     if feat != "linkWeight"]
        }
        self.feature_counts = {k: len(v) for k, v in self.acceptable_features.items()}

    # ====== Bookkeeping ============================================================

    def _print_module_hash(self):
        actor_param_hash = canonical_hash([p.data.cpu() for p in self.actor.parameters()])
        LOG_CONSOLE_INFO(f"actor hash {actor_param_hash}")
        obs_normalizer_hash = self.obs_normalizer.to_canonical_hash_str()
        LOG_CONSOLE_INFO(f"obs. normalizer hash {obs_normalizer_hash}")

    def save(self, msg="save"):
        """
        Stores the state dict of our actor as well as normalizer statistics on the disk.
        """
        if self.save_path is None:
            LOG_CONSOLE_INFO("No save path specified, model will not be saved to disk.")
        LOG_CONSOLE_RUN(f"--- saving model ({msg}) ---")
        torch.save(self.actor.state_dict(), self.save_path / f"actor.pth")
        self.obs_normalizer.save(self.save_path / f"obs_normalizer.pkl")
        self._print_module_hash()

    def load(self, load_path: Path):
        LOG_CONSOLE_INFO(f"--- loading model from {load_path} ---")
        self.actor.load_state_dict(torch.load(load_path / "actor.pth", weights_only=True))
        self.obs_normalizer = ObservationNormalizer.load(load_path / f"obs_normalizer.pkl")
        self._print_module_hash()

    # ====== Actor ===========================================================

    def _get_actor(self, nn_config: DictConfig, device: str) -> nn.Module:
        raise NotImplementedError

    @property
    def actor_params(self):
        return self.actor.parameters()

    def set_train(self, mode: bool):
        self.actor.train(mode)

    # ====== Interaction =====================================================

    def reset_env(self, scenario: NetworkScenario) -> Observation:
        """
        Resetting the env requires resetting the normalizers, too.
        """
        obs = super().reset_env(scenario)
        obs = self.obs_normalizer.reset(obs)
        return obs

    def post_step(self,
                  obs: Observation,
                  action_raw: ActionRaw,
                  action_sedi: ActionSedi,
                  logprob: LogprobTensor,
                  value: ValueTensor,
                  reward: Reward,
                  next_obs_raw: Observation,
                  done: bool,
                  infos: Dict[str, Any]) -> Observation:
        """
        Trainable algorithms require obs normalization.
        """
        next_obs = super().post_step(obs, action_raw, action_sedi, logprob, value, reward, next_obs_raw, done, infos)
        next_obs = self.obs_normalizer.normalize(next_obs)
        return next_obs
