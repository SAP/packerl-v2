from copy import deepcopy
from pathlib import Path
from typing import Dict

import networkx as nx
from omegaconf import DictConfig, OmegaConf
import torch
from torch import Tensor

from .actor_mslim import MSlimActor
from src.algorithm.trainable._trainable import TrainableAlgorithm
from src.scenario import NetworkScenario
from src.utils.observations import Observation, ObservationBatch, ObservationGraph
from src.utils.sp_calculator import apsp_to_action_sedi, sp_to_action_sedi
from src.utils.types import ActionRaw, ActionSedi


LINE_GRAPH_NODE_TO_EDGE_FEATURE_OPTIONS = {
    "none": 0,
    "inner": 1,
    "outer": 2,
    "all": 3
}  # also serve as feature count multipliers


def inv_softplus(y, beta=1.0, threshold=20.0):
    yb = y * beta
    return torch.where(yb > threshold, y, y + torch.log(-torch.expm1(-yb)) / beta)


class MSlimAlgorithm(TrainableAlgorithm):
    """
    TODO desc
    """
    name: str = "M-Slim"
    action_raw_representation = "link_weight"
    softplus = torch.nn.Softplus()
    use_line_graph: bool  # Initialized via config - determines whether to convert input graph to line graph
    line_graph_node_to_edge_features: str  # Initialized via config
    actor_out_feats: int  # Initialized in subclasses
    pred_in_flat_space: bool  # Initialized in subclasses

    def __init__(self,
                 env,
                 trainable_config: DictConfig,
                 save_path: Path | None,
                 load_path: Path | None,
                 device: str = "cpu"):
        for k, v in trainable_config.mslim.items():
            setattr(self, k, v)
        super().__init__(env, trainable_config, save_path, load_path, device)
        self._last_link_weights: Dict[int, (Tensor, Tensor)] = {}  # stored tuple of (last link weights, edge_agent_index) for each observer
        if self.pred_in_flat_space:
            self.map_link_weights = lambda x: torch.nn.functional.softplus(x)
            self.inv_map_link_weights = lambda x: inv_softplus(x)
        else:
            self.map_link_weights = lambda x: torch.exp(x)
            self.inv_map_link_weights = lambda x: torch.log(x)

    def _prepare_features(self, feat_config: DictConfig):
        super()._prepare_features(feat_config)
        self.link_weights_as_input = feat_config.link_weights_as_input
        if self.link_weights_as_input:
            self.acceptable_features["edge"].append("linkWeight")
            self.feature_counts["edge"] += 1
        self.link_weights_as_input_mode = feat_config.link_weights_as_input_mode
        self.link_weights_as_input_store_log  = feat_config.link_weights_as_input_store_log

    def _get_actor(self, nn_config: DictConfig, device: str) -> MSlimActor:
        if self.use_line_graph:
            if self.line_graph_node_to_edge_features not in LINE_GRAPH_NODE_TO_EDGE_FEATURE_OPTIONS.keys():
                raise ValueError(
                    f"Invalid option for line_graph_node_to_edge_features: {self.line_graph_node_to_edge_features}")
            edge_mult = LINE_GRAPH_NODE_TO_EDGE_FEATURE_OPTIONS[self.line_graph_node_to_edge_features]
            feat_counts = {
                "node": self.feature_counts["edge"],
                "edge": self.feature_counts["node"] * edge_mult,
                "global": self.feature_counts["global"]
            }
        else:
            feat_counts = self.feature_counts
        return MSlimActor(OmegaConf.to_container(nn_config, resolve=True),
                          feat_counts,
                          self.actor_out_feats,
                          device,
                          receives_line_graph=self.use_line_graph)

    def get_actor_view(self, obs: Observation | ObservationBatch) -> Observation | ObservationBatch:
        return obs.get_actor_input(actor_loc=self.cur_loc,
                                   actor_input=self.cur_input,
                                   link_weights_as_input=self.link_weights_as_input)

    def reset_env(self, scenario: NetworkScenario) -> Observation:
        self._last_link_weights = {}  # reset stored link weights
        obs = super().reset_env(scenario)
        return obs

    def _relevant_observer_graph_idx(self, obs: Observation) -> Tensor:
        if self.cur_loc == "single":
            if self.cur_input == "birdseye":
                relevant_observer_graph_idx = obs.birdseye_observer_graph_idx
            elif self.cur_input == "central":
                relevant_observer_graph_idx = obs.central_observer_graph_idx
            else:
                raise ValueError(f"Invalid actor_input for setting link weights")
        else:  # self.cur_actor_loc == "multi":
            relevant_observer_graph_idx = obs.local_observer_graph_idx
        return relevant_observer_graph_idx

    def set_last_link_weights(self, obs: ObservationGraph, link_weights: Tensor):
        if self.link_weights_as_input_store_log and not self.pred_in_flat_space:
            last_link_weights = torch.log(link_weights.detach().clone() + 1e-8)  # store log weights
        else:
            last_link_weights = link_weights.detach().clone()
        self._last_link_weights[obs.observer_ids.item()] = last_link_weights, obs.edge_agent_index.detach().clone()

    def _get_init_link_weight(self, num_edges: int, is_relevant: bool) -> Tensor:
        if is_relevant:
            # relevant observer: provide init link weights depending on mode
            if self.link_weights_as_input_mode == "random":
                initial_link_weights = torch.rand(num_edges, dtype=torch.float) * 9 + 1  # random weights in [1, 10]
            elif self.link_weights_as_input_mode == "ones":
                initial_link_weights = torch.ones(num_edges, dtype=torch.float)  # constant weights
            else:
                initial_link_weights = torch.zeros(num_edges, dtype=torch.float)  # default: zero weights
        else:
            # irrelevant observer: set to NaN because this should always be masked out later
            initial_link_weights = torch.full((num_edges,), float('nan'))
        return initial_link_weights

    @staticmethod
    def merge_link_weights(stored_weights, stored_edge_agent_index,
                           filler_weights, new_edge_agent_index) -> Tensor:
        # start from filler, but don't mutate the original
        merged_link_weights = filler_weights.clone()

        if stored_edge_agent_index.numel() == 0:
            return merged_link_weights  # nothing to merge

        # build a 64-bit hash key for each (u,v): key = u * B + v
        # B must be > max node id; get it from both indices to avoid collisions
        B = (torch.cat([stored_edge_agent_index, new_edge_agent_index], dim=1).max() + 1).to(torch.long)

        def make_keys(ei):
            return ei[0].to(torch.long) * B + ei[1].to(torch.long)

        old_keys = make_keys(stored_edge_agent_index)
        new_keys = make_keys(new_edge_agent_index)

        # sort old keys once, then binary-search all new keys
        order = torch.argsort(old_keys)
        old_sorted = old_keys[order]

        pos = torch.searchsorted(old_sorted, new_keys)
        pos_clamped = pos.clamp(max=old_sorted.numel() - 1)

        has_match = (pos < old_sorted.numel()) & (old_sorted[pos_clamped] == new_keys)
        if has_match.any():
            src_idx = order[pos_clamped[has_match]]
            merged_link_weights[has_match] = stored_weights[src_idx]

        return merged_link_weights

    def _get_last_link_weights(self, obs_raw: Observation) -> Tensor:

        relevant_observers = set(obs_raw.observer_ids[self._relevant_observer_graph_idx(obs_raw)].tolist())

        Es = obs_raw.num_edges_per_graph
        edge_agent_index_list = obs_raw.edge_agent_index_without_offset.split(Es.tolist(), dim=1)
        link_weights_list = []
        for observer_i, edge_agent_index_i, num_edges in zip(obs_raw.observer_ids, edge_agent_index_list, Es):
            is_relevant_observer = int(observer_i.item()) in relevant_observers
            init_link_weights_i = self._get_init_link_weight(num_edges, is_relevant_observer)
            if observer_i.item() not in self._last_link_weights:
                # no stored link weights for this observer yet -> use initial link weights
                last_link_weights_i = init_link_weights_i
            else:
                # stored link weights exist -> take existing last_link_weights
                # wherever old edge_index matches current edge_index, else use init_link_weights
                stored_link_weights_i, stored_edge_agent_index_i = self._last_link_weights[observer_i.item()]
                last_link_weights_i = self.merge_link_weights(stored_link_weights_i, stored_edge_agent_index_i,
                                                              init_link_weights_i, edge_agent_index_i)

            link_weights_list.append(last_link_weights_i)
        return torch.cat(link_weights_list).unsqueeze(-1)

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

        # last link weight as auxiliary edge feature
        if self.link_weights_as_input:
            edge_attr = torch.cat([edge_attr, self._get_last_link_weights(obs_raw)], dim=-1)

        # assign the filtered attributes back to the observation
        obs.x = node_attr
        obs.edge_attr = edge_attr
        obs.u = global_attr
        return obs

    def _get_link_weights(self, actor_input: Observation) -> Tensor:
        raise NotImplementedError

    def _get_action_sedi(self, link_weights: Tensor, obs: ObservationGraph) -> ActionSedi:
        """
        Given the link weights for the edges of the input graph, this function calculates the shortest paths for all
        node pairs and returns the indices of the edges that are part of the shortest paths, because these are our
        selected edge destination indices (sedi).
        """
        # create a NetworkX graph from the observation graph and link weights
        obs_nx = nx.DiGraph()
        obs_nx.add_nodes_from(obs.agent_ids.tolist())
        obs_nx.add_weighted_edges_from(zip(obs.edge_agent_index[0].tolist(),
                                           obs.edge_agent_index[1].tolist(),
                                           link_weights.tolist()))

        if self.cur_loc == "single":
            # single-agent case: link weights are used to compute APSP and derive sedi for all routing nodes
            apsp = nx.all_pairs_dijkstra(obs_nx)
            sedi = apsp_to_action_sedi(apsp, obs.agent_ids, obs.edge_agent_index)

        else:  # self.cur_loc == "multi"
            # multi-agent case: link weights are used to compute SSSP and derive sedi for one agent
            src_agent_id = obs.observer_ids.item()
            sssp = nx.single_source_dijkstra(obs_nx, source=src_agent_id)
            sedi = sp_to_action_sedi(sssp, src_agent_id, obs.agent_ids, obs.edge_agent_index, obs.edge_index)

        return sedi

    def _get_action(self, obs_graph: ObservationGraph) -> (ActionRaw, ActionSedi):
        """
        The PPO algorithm uses the actor-value module to get a deterministic action during validation/evaluation.
        """
        actor_input = obs_graph.clone().to(self.actor.device)
        if self.use_line_graph:
            actor_input = actor_input.to_line_graph(self.line_graph_node_to_edge_features)
        link_weights = self._get_link_weights(actor_input)
        action_sedi = self._get_action_sedi(link_weights, obs_graph)
        self.set_last_link_weights(obs_graph, link_weights.detach().clone().cpu())
        return link_weights, action_sedi


class NewMSlimAlgorithm(MSlimAlgorithm):
    actor_out_feats = 2

    min_mu = -30
    max_mu = 30
    min_log_sigma = -5.0
    max_log_sigma = 3.0
    policy_net_out_feat = 2

    def _split_actor_output(self, actor_output: Tensor) -> (Tensor, Tensor):
        mu = actor_output[:, 0].clamp(self.min_mu, self.max_mu)  # link weight log means
        log_sigma = actor_output[:, 1].clamp(self.min_log_sigma, self.max_log_sigma)  # link weight log sigmas
        sigma = log_sigma.exp()
        return mu, sigma

    def _get_link_weights(self, obs: Observation) -> Tensor:
        actor_output = self.actor(obs)
        mu, _ = self._split_actor_output(actor_output)
        return self.map_link_weights(mu)


class NewLogSpaceMSlimAlgorithm(NewMSlimAlgorithm):
    pred_in_flat_space = False


class NewFlatSpaceMSlimAlgorithm(NewMSlimAlgorithm):
    pred_in_flat_space = True


class TMLR24MSlimAlgorithm(MSlimAlgorithm):
    actor_out_feats = 1

    def _get_link_weights(self, obs: Observation) -> ActionRaw:
        actor_output = self.actor(obs).squeeze(-1)
        return self.map_link_weights(actor_output)


class TMLR24LogSpaceMSlimAlgorithm(TMLR24MSlimAlgorithm):
    pred_in_flat_space = False


class TMLR24FlatSpaceMSlimAlgorithm(TMLR24MSlimAlgorithm):
    pred_in_flat_space = True
