from copy import deepcopy
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import torch
from torch_scatter import scatter_max

from src.algorithm.trainable._nexthop_actor import NextHopActor
from .actor_dest import FieldlinesDestActor
from .actor_node_emb import (FieldLinesNodeEmbSoftBoundActor,
                             FieldLinesNodeEmbHardBoundActor,
                             FieldLinesNodeEmbCosineSimilarityActor)
from .actor_tmlr24 import FieldlinesTMLR24Actor
from src.algorithm.trainable._trainable import TrainableAlgorithm
from src.utils.observations import ObservationGraph, Observation
from src.utils.observations.obs_utils import get_nexthop_candidate_idx
from src.utils.sp_calculator import get_shortest_path_calculator
from src.utils.types import ActionRaw, ActionSedi


def get_fieldlines_actor(design: str, nn_config: dict, feature_counts: dict, device: str) -> NextHopActor:
    """
    TODO desc
    """
    if design == "new":
        return FieldlinesDestActor(nn_config, feature_counts, device)
    elif design == "tmlr24":
        return FieldlinesTMLR24Actor(nn_config, feature_counts, device)
    elif design == "node_emb_soft_bound":
        return FieldLinesNodeEmbSoftBoundActor(nn_config, feature_counts, device)
    elif design == "node_emb_hard_bound":
        return FieldLinesNodeEmbHardBoundActor(nn_config, feature_counts, device)
    elif design == "node_emb_cosine":
        return FieldLinesNodeEmbCosineSimilarityActor(nn_config, feature_counts, device)
    else:
        raise ValueError(f"Unknown Fieldlines actor design '{design}'")


class FieldlinesAlgorithm(TrainableAlgorithm):
    """
    TODO desc
    """
    name: str = "FieldLines"
    action_raw_representation = "next_hop"
    softplus = torch.nn.Softplus()
    actor_design: str  # Initialize via config
    pred_in_flat_space: bool  # NotImplemented; Implemented in subclasses

    def __init__(self,
                 env,
                 trainable_config: DictConfig,
                 save_path: Path | None,
                 load_path: Path | None,
                 device: str = "cpu"):
        for k, v in trainable_config.fieldlines.items():
            setattr(self, k, v)
        super().__init__(env, trainable_config, save_path, load_path, device)

    def _prepare_features(self, feat_config: DictConfig):
        super()._prepare_features(feat_config)
        self.spdist_as_input = feat_config.spdist_as_input
        self._spdist_provider = get_shortest_path_calculator(feat_config.spdist_mode)

    def _get_actor(self, nn_config: DictConfig, device: str) -> NextHopActor:
        return get_fieldlines_actor(self.actor_design,
                                    OmegaConf.to_container(nn_config, resolve=True),
                                    self.feature_counts,
                                    device)

    def _obtain_spdist(self, obs: Observation) -> torch.Tensor:
        """
        calculate shortest-paths distances between all nodes of the current raw observation (used as auxiliary feature)
        """
        # TODO: Currently, we supply central/birdseye node distances only, even if running in "local" multi-agent mode.
        #  This is suboptimal, since in local mode, each agent would ideally have its own node distances stemming from
        #  its local view on the network topology. This we can't compute since it would be N all-pairs shortest-path
        #  calculations per observation.
        #  Ideally, if we need spdist as feature, we should compute SSSP from each monitoringNode in ns3-packerl
        #  once the node registers a topology change, and include its node distances in the monitoringSnapshot.
        #  Then, in the assembly function, we construct the N*N spdist matrix for each node. This way, we don’t need to
        #  compute spdist in the Python side anymore.
        if self.cur_input == "birdseye" or self.cur_input == "local":
            spdist_graph = obs.get_birdseye_obs_graph()
        elif self.cur_input == "central":
            spdist_graph = obs.get_central_obs_graph()
        else:
            raise NotImplementedError("Unknown cur_input mode '{self.cur_input}'")

        spdist_numpy, _ = self._spdist_provider.get_node_distances_and_central_node(spdist_graph)
        return torch.from_numpy(spdist_numpy[:, 2])  # shape: [N**2,]

    def _prepare_obs(self, obs_raw: Observation) -> Observation:
        """
        Filters out irrelevant features
        """
        # clone the input observation to avoid modifying it
        obs = deepcopy(obs_raw)

        # filter out unwanted features
        node_attr = obs.x[:, self.acceptable_feat_idx["node"]]
        edge_attr = obs.edge_attr[:, self.acceptable_feat_idx["edge"]]
        global_attr = obs.u[:, self.acceptable_feat_idx["global"]]

        # - shortest-path distances as feature
        if self.spdist_as_input:
            obs.__setattr__("spdist", self._obtain_spdist(obs_raw))

        # assign the filtered attributes back to the observation
        obs.x = node_attr
        obs.edge_attr = edge_attr
        obs.u = global_attr

        return obs

    def _get_action(self, obs_graph: ObservationGraph) -> (ActionRaw, ActionSedi):
        """
        TODO desc
        """
        edge_dest_scores = self.actor(obs_graph.clone().to(self.actor.device))  # 1-dim tensor, numel=sum([g.E*g.N for graph g in batch])
        _, action_sedi = scatter_max(edge_dest_scores, index=get_nexthop_candidate_idx(obs_graph))  # edge_dest_values, action_sedi
        return edge_dest_scores, action_sedi
