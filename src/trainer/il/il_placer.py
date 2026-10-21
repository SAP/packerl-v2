import networkx as nx
import numpy as np
from omegaconf import DictConfig
import torch
from torch import Tensor

from ._il import ImitationLearningTrainer
from ._supervised_losses import dist_matrix_loss, sedi_sparse_ce_loss
from src.algorithm.trainable import TrainableAlgorithm
from src.algorithm.trainable.placer import PlacerAlgorithm
from src.packerl_env import PackerlEnv
from src.utils.observations import Observation, ObservationGraph


def link_weights_to_pairwise_dist(link_weights: Tensor, ref_graph: ObservationGraph) -> Tensor:
    """
    TODO integrate into SP calculator
    """
    ei = ref_graph.edge_index
    N = ref_graph.num_nodes

    # create a weighted directed graph
    weighted_graph = nx.DiGraph()
    weighted_graph.add_nodes_from(range(N))
    weighted_edges = [(u.item(), v.item(), w.item()) for u, v, w in zip(ei[0], ei[1], link_weights)]
    weighted_graph.add_weighted_edges_from(weighted_edges)

    # obtain all-pairs shortest paths
    apsp = nx.all_pairs_dijkstra(weighted_graph)

    # transform to distance matrix
    node_dist_arr = np.empty((N ** 2, 3), dtype=np.float32)  # shape: (N^2, 3)
    i = 0
    for src_node, (distances, _) in apsp:
        for dst_node, dist in distances.items():
            node_dist_arr[i] = (src_node, dst_node, dist)
            i += 1
    node_dist = node_dist_arr[np.lexsort((node_dist_arr[:, 1], node_dist_arr[:, 0]))]  # sort rows by src, then dst
    return torch.from_numpy(node_dist[:, 2]).view(N, N)  # shape: (N, N)


class ILPlacerTrainer(ImitationLearningTrainer):

    geom_loss_weight: float  # weight of the geometric loss component, set by config

    def __init__(self, config: DictConfig, trainer_config: DictConfig, env: PackerlEnv, algorithm: TrainableAlgorithm):
        super().__init__(config, trainer_config, env, algorithm)

        if self.expert.action_raw_representation == "next_hop":
            raise ValueError("Placer student can't be trained with next_hop expert demonstrations")

    def _get_supervised_loss(self,
                             action_raw: Tensor,
                             action_sedi_sparse: Tensor,
                             expert_action_raw: Tensor,
                             expert_sedi_sparse: Tensor,
                             input_obs: Observation) -> Tensor:
        """
        The behavior cloning loss for the placer task is a cross-entropy loss
        """
        assert isinstance(self._algo, PlacerAlgorithm)
        node_emb = action_raw
        edge_scores = self._algo.node_emb_to_edge_scores(node_emb, input_obs)

        # component 1: cross-entropy loss on the sedi distribution
        ce_loss = sedi_sparse_ce_loss(student_sedi_soft_sparse=edge_scores.view(-1),
                                      expert_sedi_onehot_sparse=expert_sedi_sparse.to(edge_scores.device),
                                      edge2src=input_obs.edge_index[0].to(edge_scores.device)
                                      )
        if self.geom_loss_weight <= 0:
            return ce_loss

        # component 2: aux. geometric loss via pairwise shortest distances (computed separately for each input graph)
        Ns = input_obs.num_nodes_per_graph.tolist()
        Es = input_obs.num_edges_per_graph.tolist()
        input_graphs = input_obs.to_obs_graph_list()
        stud_ne_per_graph = node_emb.split(Ns, dim=0)
        expert_out = expert_action_raw.to(node_emb.device)
        if self.expert.action_raw_representation == "link_weight":
            exp_out_per_graph = expert_out.split(Es, dim=0)
        elif self.expert.action_raw_representation == "node_emb":
            assert isinstance(self.expert, PlacerAlgorithm)
            exp_out_per_graph = expert_out.split(Ns, dim=0)
        else:
            raise ValueError(f"Invalid expert action representation {self.expert.action_raw_representation}")

        geom_losses = []
        for g, stud_ne_g, exp_out_g in zip(input_graphs, stud_ne_per_graph, exp_out_per_graph):
            # student "pairwise distances" are normally treated as squared distances -> sqrt to get actual distances
            stud_pairwise_dists = torch.sqrt(self._algo.get_pairwise_distances(stud_ne_g) + 1e-8)  # [n, n]
            if self.expert.action_raw_representation == "link_weight":
                exp_pairwise_dists = link_weights_to_pairwise_dist(exp_out_g, g)  # [n, n]
            elif self.expert.action_raw_representation == "node_emb":
                exp_pairwise_dists = self.expert.get_pairwise_distances(exp_out_g)  # [n, n]
            else:
                raise ValueError(f"Invalid expert action representation {self.expert.action_raw_representation}")
            geom_losses.append(dist_matrix_loss(stud_pairwise_dists, exp_pairwise_dists))
        geom_loss = torch.stack(geom_losses).mean()

        # put it together
        return ce_loss + self.geom_loss_weight * geom_loss
