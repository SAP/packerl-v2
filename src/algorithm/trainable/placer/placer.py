from copy import deepcopy
import math
from pathlib import Path
from typing import Callable

from omegaconf import DictConfig, OmegaConf
import torch
from torch import Tensor, nn
from torch_scatter import scatter_max

from src.algorithm.trainable._trainable import TrainableAlgorithm
from src.algorithm.trainable.placer.actor_placer import PlacerActor
from src.utils.observations import ObservationGraph, Observation, ObservationBatch
from src.utils.observations.obs_utils import get_nexthop_candidate_idx
from src.utils.sp_calculator import get_shortest_path_calculator
from src.utils.types import ActionRaw, ActionSedi


NODE_EMB_TAU = 2.0  # Initial temperature for node-embedding-based actor
NODE_EMB_TAU_BOUNDS = (0.5, 4.0)  # Bounds for temperature tau


def pairwise_hard_bound_dist(node_emb: Tensor) -> Tensor:
    """
    This variant does not bound the embedding radii, but instead simply clamps the final squared distances.
    """
    squared_norms = torch.sum(node_emb ** 2, dim=1, keepdim=True)
    pairwise_squared_dists = squared_norms + squared_norms.t() - 2 * torch.matmul(node_emb, node_emb.t())
    return pairwise_squared_dists.clamp_(min=0.0, max=4.0)


def pairwise_soft_bound_dist(node_emb: Tensor, R: float = 1.0, s: float = 1.0, eps : float = 1e-8) -> Tensor:
    """
    This variant uses a tanh to "softly" bound the radii of the embeddings prior to compute pairwise squared dists.
    It presents a middle ground between the two other variants.

    Args:
        node_emb: Node embeddings, shape [num_nodes, embedding_dim]
        R: Maximum radius for the embeddings after soft-bounding
        s: Scaling factor for the soft-bounding
        eps: Small value to prevent division by zero
    """
    norms = node_emb.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    direction = node_emb / norms  # directions
    radius = R * torch.tanh(norms / s)  # smooth, bounded radii, values in [0, R)
    cosine_similarity = direction @ direction.t()  # values in [-1, 1]
    sq_radius = radius.pow(2)  # values in [0, R^2)
    pairwise_squared_dists = sq_radius + sq_radius.t() - 2.0 * (radius @ radius.t()) * cosine_similarity
    return pairwise_squared_dists.clamp_(min=0.0, max=(2 * R) ** 2)


def pairwise_cosine_similarity(node_emb: Tensor, eps: float = 1e-8) -> Tensor:
    norms = node_emb.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    direction = node_emb / norms
    cosine_similarity = direction @ direction.t()  # values in [-1, 1]
    # Squared Euclid on unit sphere in [0, 4] via cosine similarity
    pairwise_dist = 2 * (1 - cosine_similarity)
    return pairwise_dist.clamp_(min=0.0, max=4.0)


class PlacerAlgorithm(TrainableAlgorithm):
    """
    TODO desc
    """
    name: str = "Placer"
    action_raw_representation = "node_emb"
    embedding_dim: int  # dimension of node embeddings
    distance_mode: str  # one of "hb", "sb", "cs"
    get_pairwise_distances: Callable  # function to compute pairwise distances between node embeddings

    def __init__(self,
                 env,
                 trainable_config: DictConfig,
                 save_path: Path | None,
                 load_path: Path | None,
                 device: str = "cpu"):
        for k, v in trainable_config.placer.items():
            setattr(self, k, v)
        super().__init__(env, trainable_config, save_path, load_path, device)

        self._get_pairwise_dist_func()
        self.log_tau = nn.Parameter(torch.tensor(math.log(NODE_EMB_TAU), device=device))

    def _prepare_features(self, feat_config: DictConfig):
        super()._prepare_features(feat_config)
        self.spdist_as_input = feat_config.spdist_as_input
        self._spdist_provider = get_shortest_path_calculator(feat_config.spdist_mode)

    def _get_actor(self, nn_config: DictConfig, device: str) -> PlacerActor:
        return PlacerActor(OmegaConf.to_container(nn_config, resolve=True),
                           self.feature_counts,
                           self.embedding_dim,
                           device)

    def _get_pairwise_dist_func(self):
        if self.distance_mode == "sb":
            self.get_pairwise_distances = pairwise_hard_bound_dist
        elif self.distance_mode == "cs":
            self.get_pairwise_distances = pairwise_cosine_similarity
        elif self.distance_mode == "hb":
            self.get_pairwise_distances = pairwise_soft_bound_dist
        else:
            raise ValueError(f"Unknown distance mode {self.distance_mode}")

    def tau(self) -> Tensor:
        # Positive and bounded τ; prevents extreme logit gaps.
        t = self.log_tau.exp()
        return t.clamp(NODE_EMB_TAU_BOUNDS[0], NODE_EMB_TAU_BOUNDS[1])

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

    def node_emb_to_edge_scores(self,
                                node_emb: Tensor,
                                obs: ObservationGraph | ObservationBatch) -> Tensor:
        """
        TODO desc
        """
        if isinstance(obs, ObservationBatch):
            Ns = obs.num_nodes_per_graph.tolist()  # list of num nodes per graph in the batch
            Es = obs.num_edges_per_graph.tolist()  # list of num edges per graph in the batch
            node_emb_per_graph = node_emb.split(Ns, dim=0)
            edge_dests_per_graph = obs.edge_index_without_offset[1].split(Es)
        else:
            Ns = [obs.num_nodes]
            Es = [obs.num_edges]
            node_emb_per_graph = [node_emb]
            edge_dests_per_graph = [obs.edge_index[1]]

        tau = self.tau()
        edge_dest_scores = []

        for n, e, node_emb_g, edge_dests_g in zip(Ns, Es, node_emb_per_graph, edge_dests_per_graph):
            # pairwise distances between all nodes in the graph
            pairwise_dists = self.get_pairwise_distances(node_emb_g)
            pairwise_logits = -pairwise_dists / tau

            edge_destinations = edge_dests_g.repeat_interleave(n, dim=0)
            traffic_destinations = torch.arange(n, device=self.actor.device).repeat(e)
            edge_dest_scores.append(pairwise_logits[edge_destinations, traffic_destinations])  # 1-dim tensor, numel=e*n

        return torch.cat(edge_dest_scores, dim=0)  # 1-dim tensor, numel=sum([g.E*g.N for graph g in batch])

    def _get_action(self, obs_graph: ObservationGraph) -> (ActionRaw, ActionSedi):
        """
        TODO desc
        """
        node_emb = self.actor(obs_graph.clone().to(self.actor.device))  # [num_nodes, embedding_dim]
        edge_dest_scores = self.node_emb_to_edge_scores(node_emb, obs_graph.to(node_emb.device))  # [num_edges * num_nodes,]
        _, action_sedi = scatter_max(edge_dest_scores, index=get_nexthop_candidate_idx(obs_graph))
        return node_emb, action_sedi
