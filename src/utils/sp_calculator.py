from itertools import tee
from typing import Tuple

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_scatter import scatter_max

from src.utils.actions.ac_utils import sp_actions_to_ascii
from src.utils.constants import (BIRDSEYE_OBSERVER_ID,
                                 OSPF_REF_DATARATE,
                                 EIGRP_REF_DATARATE,
                                 EIGRP_REF_DELAY,
                                 EIGRP_REF_MULTIPLIER)
from src.utils.features import edge_feat_in_struct_order, edge_datarate_idx, edge_delay_idx
from src.utils.logging import LOG_CONSOLE_INFO, LOG_CONSOLE_FUNCTION
from src.utils.observations import ObservationGraph, Observation
from src.utils.types import ActionRaw, ActionSedi


def get_shortest_path_calculator(sp_mode: str) -> 'ShortestPathCalculator':
    """
    Factory function to create a ShortestPathCalculator instance based on the given sp_mode.
    """
    if sp_mode == "ospf":
        return OSPFShortestPathCalculator()
    elif sp_mode == "eigrp":
        return EIGRPShortestPathCalculator()
    elif sp_mode == "delay":
        return DelayShortestPathCalculator()
    elif sp_mode == "hop":
        return HopCountShortestPathCalculator()
    else:
        raise ValueError(f"Invalid sp_mode: {sp_mode}, can't create ShortestPathCalculator")


def apsp_to_node_dist(apsp, num_nodes: int) -> (np.ndarray, int):
    """
    Convert the all-pairs shortest paths (APSP) result into a node distance array and find the central node
     (the node with the minimum maximum distance to all other nodes).

    Args:
        apsp: All-pairs shortest paths result from nx, which is a generator of tuples (src_node, (distances, paths)),
        num_nodes: int, number of nodes in the graph.
    """

    node_dist_arr = np.empty((num_nodes ** 2, 3), dtype=np.float32)  # shape: (N^2, 3)
    i = 0
    for src_node, (distances, _) in apsp:
        for dst_node, dist in distances.items():
            node_dist_arr[i] = (src_node, dst_node, dist)
            i += 1

    # Sort distances; get central node (node with minimum maximum distance to all other nodes)
    node_dist = node_dist_arr[np.lexsort((node_dist_arr[:, 1], node_dist_arr[:, 0]))]  # sort rows by src, then dst
    node_dist_np_sq = torch.from_numpy(node_dist[:, 2]).view(num_nodes, num_nodes)
    max_dist, _ = node_dist_np_sq.max(dim=1)
    central_node_idx = max_dist.argmin().item()  # index of node with minimum maximum distance to all other nodes
    central_node = node_dist[central_node_idx * num_nodes, 0].astype(np.int64).item()
    return node_dist, central_node


def sp_to_action_sedi(sp,
                      src_agent_id: int,
                      agent_ids: Tensor,
                      edge_agent_index: Tensor,
                      inner_edge_index: Tensor,
                      debug: bool=False) -> ActionSedi:
    """
    Convert a single-source shortest path (SSSP) result into a selected edge destination indices (sedi) tensor.

    While the SSSP result is in the outer agent ID representation (like edge_agent_index), the filling of
    sp_action and action_sedi is done in the inner node ID/index representation.

    the agent_ids are a tensor of shape [N,] containing the agent IDs for each node in the graph.
    the edge_agent_index is a tensor of shape [2, E] containing the source and destination agent IDs for each edge.
    """
    num_nodes = agent_ids.size(0)  # number of nodes in the graph
    num_edges = edge_agent_index.shape[1]
    flat_idx_multiplier = edge_agent_index.max().item() + 1
    edge_lookup = torch.full((flat_idx_multiplier, flat_idx_multiplier), -1,
                             dtype=torch.long, device=edge_agent_index.device)
    rows = torch.arange(num_edges, dtype=torch.int64, device=edge_agent_index.device)
    edge_lookup[edge_agent_index[0], edge_agent_index[1]] = rows

    # create fully random sparse action tensor of shape [E, N] (edges x nodes).
    sp_action = torch.rand((num_edges, num_nodes))
    # fill in shortest path actions for src_agent_id:
    # - for each dst_agent_id, find the next-hop edge from src_agent_id and its value in random_sp_actions to 1.
    # - scatter_max will thus select these edges as next-hop edges, and random next-hop edges for all other agents.
    for dst_agent_id, path in sp[1].items():
        if src_agent_id == dst_agent_id: continue  # simply leave the random edge as is if src == dst
        i = int(edge_lookup[src_agent_id, int(path[1])].item())  # index of next-hop edge from agent
        j = (agent_ids == dst_agent_id).nonzero(as_tuple=False).squeeze()  # index of dst_agent_id in inner node IDs
        sp_action[i, j] = 1
    sp_action = sp_action.view(-1)  # flatten to 1D tensor of shape [E*N,]

    if debug:
        print(sp_actions_to_ascii(sp_action, num_nodes, edge_agent_index))

    inner_edge_src = inner_edge_index[0].unsqueeze(-1).expand(-1, num_nodes)  # shape: (E, N)
    dest_node_range = torch.arange(num_nodes).unsqueeze(0).expand(num_edges, -1)  # shape: (E, N)
    nexthop_dest_idx = (inner_edge_src * num_nodes + dest_node_range).view(-1)  # shape: (E * N)
    _, action_sedi = scatter_max(sp_action, index=nexthop_dest_idx)
    return action_sedi


def apsp_to_action_sedi(apsp, agent_ids: Tensor, edge_agent_index: Tensor, debug: bool=False) -> ActionSedi:
    """
    Convert the all-pairs shortest paths (APSP) result into a selected edge destination indices (sedi) tensor.

    While the APSP result is in the outer agent ID representation (like edge_agent_index), the filling of
    sp_action and action_sedi is done in the inner node ID/index representation.

    the agent_ids are a tensor of shape [N,] containing the agent IDs for each node in the graph.
    the edge_agent_index is a tensor of shape [2, E] containing the source and destination agent IDs for each edge.
    """
    num_nodes = agent_ids.size(0)  # number of nodes in the graph
    num_edges = edge_agent_index.shape[1]
    flat_idx_multiplier = edge_agent_index.max().item() + 1
    edge_lookup = torch.full((flat_idx_multiplier, flat_idx_multiplier), -1,
                             dtype=torch.long, device=edge_agent_index.device)
    rows = torch.arange(num_edges, dtype=torch.int64, device=edge_agent_index.device)
    edge_lookup[edge_agent_index[0], edge_agent_index[1]] = rows

    # create a sparse action tensor of shape [E, N] (edges x nodes) to store the shortest path actions.
    sp_action = torch.zeros((num_edges, num_nodes), dtype=torch.bool, device=edge_agent_index.device)
    for src_agent_id, (_, shortest_paths_per_src_node) in apsp:
        for dst_agent_id, path in shortest_paths_per_src_node.items():
            if src_agent_id == dst_agent_id: continue  # no need to fill self edges here, we do it later randomly for completeness
            i = int(edge_lookup[src_agent_id, int(path[1])].item())  # index of next-hop edge from src_agent_id
            j = (agent_ids == dst_agent_id).nonzero(as_tuple=False).squeeze()  # index of dst_agent_id in inner node IDs
            sp_action[i, j] = True
    sp_action = sp_action.view(-1)  # flatten to 1D tensor of shape [E*N,]

    # fill randomly selected self edges for completeness
    edge_agent_src_index = edge_agent_index[0]  # source agent IDs for each edge
    inner_edge_src_index = torch.tensor([(agent_ids == ag_id).nonzero(as_tuple=False).squeeze()
                                         for ag_id in edge_agent_src_index.tolist()],
                                        dtype=torch.long, device=edge_agent_index.device)
    _, rng_self_idx = scatter_max(torch.rand(num_edges, device=edge_agent_index.device),
                                                inner_edge_src_index, dim_size=num_nodes)
    if debug:
        print(sp_actions_to_ascii(sp_action, num_nodes, edge_agent_index, rng_self_idx))
    sp_action[rng_self_idx * num_nodes + torch.arange(num_nodes, device=rng_self_idx.device)] = 1

    # obtain action sedi as a 1D tensor of shape [N*N,], containing the selected edge destination indices
    action_sedi = sp_action.nonzero(as_tuple=False).squeeze()

    return action_sedi


class ShortestPathCalculator:
    """
    Class to calculate shortest paths and, from there, node distances and routing actions for a given network graph.
    Can be configured to use different ways of computing link weights, and caches the results for efficiency.
    """
    sp_mode: str

    def __init__(self):
        self.agent_ids: Tensor | None = None  # mapping from inner/PyG node IDs to outer agent IDs. shape: (num_nodes,)
        self.edge_agent_index: Tensor | None = None  # shape: (2, num_edges)
        self.edge_weights: Tensor | None = None  # shape: (num_edges,)
        self.weight_name = f"{self.sp_mode}Weight"

        self.node_distances = None
        self.action_sedi = None

        LOG_CONSOLE_INFO(f"initialized ShortestPathCalculator in mode: {self.sp_mode}")

    def _calculate_weights(self, edge_attr: Tensor) -> Tensor:
        """
        Calculate and return routing weights for given network graph.
        """
        raise NotImplementedError

    def _compute_apsp(self,
                      agent_ids: Tensor,
                      edge_agent_index: Tensor,
                      edge_weights: Tensor,
                      compute_sedi: bool = False,
                      debug: bool = False) -> (np.ndarray, int, ActionSedi | None):
        """
        Recompute the shortest paths and node distances for the given network graph.
        """
        LOG_CONSOLE_FUNCTION()

        # create weighted graph from node ids, edge index and edge weights (directed is as fast as undirected)
        weighted_graph = nx.DiGraph()
        weighted_graph.add_nodes_from(agent_ids.numpy())
        edges_with_weights = [(int(u), int(v), {self.weight_name: float(w)})
                              for (u, v), w in zip(edge_agent_index.T.numpy(), edge_weights.numpy())]
        weighted_graph.add_edges_from(edges_with_weights)

        # solve all-pairs shortest paths using Dijkstra's algorithm
        apsp = nx.all_pairs_dijkstra(weighted_graph, cutoff=None, weight=self.weight_name)

        # get the node distances and the central node, and optionally the action sedi
        num_nodes = agent_ids.size(0)
        if compute_sedi:
            apsp1, apsp2 = tee(apsp)
            node_dist, central_node = apsp_to_node_dist(apsp1, num_nodes)
            action_sedi = apsp_to_action_sedi(apsp2, agent_ids, edge_agent_index, debug)
        else:
            node_dist, central_node = apsp_to_node_dist(apsp, num_nodes)
            action_sedi = None

        return node_dist, central_node, action_sedi

    def _check_reset(self,
                     graph_data: ObservationGraph | Observation,
                     force_reset: bool,
                     require_sedi: bool = False,
                     debug: bool = False) -> None:
        """
        Check if the given network graph is different from the cached one,
        and reset the cached values if necessary.
        """
        LOG_CONSOLE_FUNCTION()

        new_edge_attr = graph_data.edge_attr
        if new_edge_attr.shape[1] != len(edge_feat_in_struct_order):
            raise ValueError("Edge attributes do not match the expected structure.")

        if isinstance(graph_data, Observation):
            new_agent_ids = graph_data.agent_ids_with_offset
            new_edge_agent_index = graph_data.edge_agent_index_with_offset
        else:
            new_agent_ids = graph_data.agent_ids
            new_edge_agent_index = graph_data.edge_agent_index
        new_weights = self._calculate_weights(new_edge_attr).float()

        needs_recompute = (force_reset
                           or self.agent_ids is None
                           or self.edge_agent_index is None
                           or self.edge_weights is None
                           or not torch.equal(self.agent_ids, new_agent_ids)
                           or not torch.equal(self.edge_agent_index, new_edge_agent_index)
                           or not torch.equal(self.edge_weights, new_weights)
                           or self.node_distances is None
                           or (require_sedi and self.action_sedi is None)
                           )

        # if no recompute is needed, we can return here
        if not needs_recompute:
            return

        # update cached topology information
        self.agent_ids = new_agent_ids  # shape: (num_nodes,)
        self.edge_agent_index = new_edge_agent_index  # shape: (2, num_edges)
        self.edge_weights = new_weights  # shape: (num_edges,)

        # recompute shortest paths and node distances
        if isinstance(graph_data, Observation):
            # if the graph data is a batch of graphs, we need to split the node ids and edge indices and compute
            # the shortest paths for each graph separately.
            new_agent_ids_split = torch.split(new_agent_ids, graph_data.num_nodes_per_graph.tolist())
            new_edge_agent_index_split = torch.split(new_edge_agent_index, graph_data.num_edges_per_graph.tolist(),
                                                     dim=1)
            new_weights_split = torch.split(new_weights, graph_data.num_edges_per_graph.tolist())
            node_distances = []
            actions_sedi = []
            action_sedi_offset = 0
            for agent_ids, edge_agent_index, edge_weights, observer_id in zip(new_agent_ids_split,
                                                                       new_edge_agent_index_split,
                                                                       new_weights_split,
                                                                       graph_data.observer_ids):
                nd, cn, ac_sedi = self._compute_apsp(agent_ids, edge_agent_index, edge_weights, require_sedi,
                                                     debug=debug)
                node_distances.append(nd)
                # if sedi are required, we need to offset the action sedi by the current action_sedi_offset
                if require_sedi:
                    # action_sedi is a 1D tensor of shape [N*N,] containing the selected edge destination indices
                    # for the current graph, we need to offset it by the current action_sedi_offset
                    ac_sedi += action_sedi_offset
                    action_sedi_offset += agent_ids.size(0) * edge_agent_index.size(1)
                actions_sedi.append(ac_sedi)

                # central nodes as obtained from `_compute_apsp` come with the agent ID offset, since we calculate them
                # from the resulting node_distances array (and those are computed for agent IDs with offsets).
                # Since we however get the central node ID from the birdseye observer, and the birdseye view
                # is put into the observation first, it comes without any offset, and we can use it directly.
                if observer_id == BIRDSEYE_OBSERVER_ID:
                    self.central_node = cn

            # Since for full 'Observation' instances the stored 'agent_ids' are with offset,
            # these node distances also contain the offset! In practice, this is not a problem because this only happens
            # when PPO uses this class to get overriding sp_actions, in which case only the resulting action sedi
            # are used (and these require the offset anyway).
            self.node_distances = np.concatenate(node_distances, axis=0)  # shape: (sum[N_i^2 for graph i in observation], 3)
            if require_sedi:
                self.action_sedi = torch.cat(actions_sedi)  # shape: (sum[N_i^2 for i in batch],)
        else:
            # if the graph data is a single graph, we can compute the shortest paths and node distances directly
            self.node_distances, self.central_node, self.action_sedi = self._compute_apsp(new_agent_ids,
                                                                                          new_edge_agent_index,
                                                                                          new_weights,
                                                                                          require_sedi,
                                                                                          debug=debug)

    def get_node_distances_and_central_node(self,
                                            graph_data: ObservationGraph,
                                            force_reset: bool = False) -> (np.ndarray, int):
        """
        Return {cached/calculated} node distances (incl. src and dst nodes in numpy array) and the central node,
         which is the node with the lowest maximum delay to all other nodes.
        """
        LOG_CONSOLE_FUNCTION()
        self._check_reset(graph_data, force_reset)
        return self.node_distances, self.central_node

    def get_node_distances_single(self,
                                  graph_data: ObservationGraph,
                                  agent_id: int,
                                  force_reset: bool = False) -> np.ndarray:
        """
        Return {cached/calculated} node distances for a single agent in the graph.
        """
        LOG_CONSOLE_FUNCTION()
        self._check_reset(graph_data, force_reset)

        agent_id_position = np.where(self.agent_ids == agent_id)[0]
        if agent_id_position.size == 0:
            raise ValueError(f"Agent ID {agent_id} not found in the graph's agent IDs.")
        if len(agent_id_position) > 1:
            raise ValueError(f"Agent ID {agent_id} found multiple times in the graph's agent IDs.")

        start_idx = agent_id_position[0] * len(self.agent_ids)
        end_idx = start_idx + len(self.agent_ids)
        return self.node_distances[start_idx:end_idx, 2].astype(np.float32)

    def get_link_weights_and_sedi(self,
                                  graph_data: ObservationGraph,
                                  force_reset: bool = False,
                                  debug: bool = False) -> Tuple[ActionRaw, ActionSedi]:
        """
        Return {cached/calculated} shortest-path weights and the computed selected edge destination indices.
        """
        LOG_CONSOLE_FUNCTION()
        self._check_reset(graph_data, force_reset, require_sedi=True, debug=debug)
        return self.edge_weights, self.action_sedi


class OSPFShortestPathCalculator(ShortestPathCalculator):
    """
    OSPF routing calculator, using the OSPF reference datarate as a weight.
    """
    sp_mode = "ospf"
    def _calculate_weights(self, edge_attr: Tensor) -> Tensor:
        """
        Calculate and return OSPF weights for given edge attributes,
        where datarate is in bps.
        """
        return OSPF_REF_DATARATE / edge_attr[:, edge_datarate_idx]


class EIGRPShortestPathCalculator(ShortestPathCalculator):
    """
    EIGRP routing calculator, using the EIGRP reference datarate and delay as weights.
    """
    sp_mode = "eigrp"
    def _calculate_weights(self, edge_attr: Tensor) -> Tensor:
        """
        Calculate and return EIGRP weights for given edge attributes,
        where datarate is converted to kbps (*=0.001 from bps),
        and delay is converted to tens of µs (*=100 from ms).
        """
        datarate_weight = EIGRP_REF_DATARATE / (edge_attr[:, edge_datarate_idx] * 0.001)  # shape: (num_edges,)
        delay_weight = 100 * edge_attr[:, edge_delay_idx] / EIGRP_REF_DELAY  # shape: (num_edges,)
        return EIGRP_REF_MULTIPLIER * (datarate_weight + delay_weight)


class DelayShortestPathCalculator(ShortestPathCalculator):
    """
    Delay-based shortest path calculator, using the channel delay as the weight.
    """
    sp_mode = "delay"
    def _calculate_weights(self, edge_attr: Tensor) -> Tensor:
        """
        Calculate and return delay weights for given edge attributes.
        """
        return edge_attr[:, edge_delay_idx]


class HopCountShortestPathCalculator(ShortestPathCalculator):
    """
    Hop-count-based shortest path calculator, using a constant weight of 1 for each link.
    """
    sp_mode = "hop"
    def _calculate_weights(self, edge_attr: Tensor) -> Tensor:
        """
        Calculate and return hop-count weights for given edge attributes.
        """
        num_edges = edge_attr.size(0)
        return torch.ones((num_edges,), dtype=torch.float32, device=edge_attr.device)