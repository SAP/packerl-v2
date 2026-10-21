"""
This module provides functions to convert between python-side components and ctypes-compatible representations
as obtained/used by the C++ side of PackeRL.
"""
from typing import Dict

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import torch

from src.scenario.events import Event as PSEvent, LinkFailure as PSLinkFailure
from src.scenario.events.traffic.flow_arrival import FlowArrival as PSFlowArrival
from src.utils.actions import PackerlEnvAction
from src.utils.constants import BIRDSEYE_OBSERVER_ID
from src.utils.shared_memory.structs import (TopologyEdge,
                                             TrafficFlowArrival as CTypeFlowArrival,
                                             LinkFailure as CTypeLinkFailure,
                                             GlobalRewardWeights,
                                             global_meta_idx,
                                             node_meta_idx,
                                             edge_meta_idx,
                                             reward_meta_idx,
                                             routing_action_dtype,
                                             routing_action_node_delay_dtype
                                             )
from src.utils.observations import Observation, ObservationGraph
from src.utils.sp_calculator import ShortestPathCalculator


def topology_edge_to_ctype(fst, snd, attributes: dict) -> TopologyEdge:
    """
    Converts a given topology edge into a ctype-compatible representation.
    """
    c_edge = TopologyEdge()
    c_edge.fst = fst
    c_edge.snd = snd
    c_edge.datarate = int(attributes["datarate"])
    c_edge.delay = int(attributes["delay"])
    return c_edge


def link_failure_to_ctype(link_failure: PSLinkFailure) -> CTypeLinkFailure:
    """
    Converts a given link failure event (from scenarios) into a ctype-compatible representation.
    """
    c_link_failure = CTypeLinkFailure()
    c_link_failure.t = link_failure.t
    c_link_failure.fst = link_failure.fst
    c_link_failure.snd = link_failure.snd
    return c_link_failure


def traffic_flow_arrival_to_ctype(arrival: PSFlowArrival) -> CTypeFlowArrival:
    """
    Converts a given traffic flow arrival event (from scenarios) into a ctype-compatible representation.
    """
    c_arrival = CTypeFlowArrival()
    c_arrival.t = arrival.t
    c_arrival.src = arrival.src
    c_arrival.dst = arrival.dst
    c_arrival.amount = arrival.amount
    c_arrival.datarate = arrival.datarate
    c_arrival.isTCP = arrival.is_tcp
    return c_arrival


def event_to_ctype(event: PSEvent):
    """
    Converts a given event (from scenarios) into a ctype-compatible representation.
    """
    if isinstance(event, PSFlowArrival):
        return 0, traffic_flow_arrival_to_ctype(event)
    elif isinstance(event, PSLinkFailure):
        return 1, link_failure_to_ctype(event)
    else:
        raise ValueError(f"Event type {type(event)} not supported!")


def actions_to_bytes(actions: PackerlEnvAction) -> (bytes, int):
    """
    Returns: Action components as bytes and the number of action components.
    """
    # Prepare data for the structured array
    num_edges, num_nodes = actions.edge_attr.shape
    edge_attr_np = actions.edge_attr.cpu().numpy().astype(np.float32)  # shape: (num_edges, num_nodes)
    edge_agent_index_np = actions.agent_ids[actions.edge_index].cpu().numpy().astype(np.int32)  # shape: (num_edges, num_nodes)

    # Create per-entry indices:
    # For each edge, there are `num_nodes` entries. So we want:
    # - edgeSrc: each source node repeated num_nodes times
    # - edgeDst: each destination node (for that edge) repeated num_nodes times
    # - destNode: 0, 1, …, num_nodes-1 repeated for each edge
    edge_src = np.repeat(edge_agent_index_np[0], num_nodes)  # shape: (num_edges * num_nodes,)
    edge_dst = np.repeat(edge_agent_index_np[1], num_nodes)  # shape: (num_edges * num_nodes,)
    dest_nodes = np.tile(np.arange(num_nodes, dtype=np.int32), num_edges)  # shape: (num_edges * num_nodes,)

    # Flatten the edge attributes to match the ordering of the edge indices
    values = edge_attr_np.flatten()  # shape: (num_edges * num_nodes,)

    # Allocate the structured array directly and fill the fields
    action_components_structured = np.empty(num_edges * num_nodes, dtype=routing_action_dtype)
    action_components_structured['edgeSrc'] = edge_src
    action_components_structured['edgeDst'] = edge_dst
    action_components_structured['trafficDst'] = dest_nodes
    action_components_structured['value'] = values

    return action_components_structured.tobytes(), len(action_components_structured)


def agent_delays_to_bytes(agent_ids: torch.Tensor, delays_per_agent: torch.Tensor) -> (bytes, int):
    """
    Converts a given set of action delays per node into a bytes representation.
    """
    num_agents = agent_ids.shape[0]
    action_delays_structured = np.empty(num_agents, dtype=routing_action_node_delay_dtype)
    action_delays_structured['nodeId'] = agent_ids.cpu().numpy().astype(np.int32)
    action_delays_structured['delay'] = delays_per_agent.cpu().numpy().astype(np.float32)
    return action_delays_structured.tobytes(), num_agents


def global_reward_weights_to_ctype(weights: Dict[str, float]) -> GlobalRewardWeights:
    """
    Converts a given set of global reward weights into a ctype-compatible representation.
    """
    g_weights = GlobalRewardWeights()

    g_weights.txWeight = weights.get('tx', 0.0)
    g_weights.reTxWeight = weights.get('reTx', 0.0)
    g_weights.inFlightWeight = weights.get('inFlight', 0.0)
    g_weights.rxWeight = weights.get('rx', 0.0)
    g_weights.deliveredWeight = weights.get('delivered', 0.0)
    g_weights.ttlDroppedWeight = weights.get('ttlDropped', 0.0)
    g_weights.tcpDiscardedWeight = weights.get('tcpDiscarded', 0.0)
    g_weights.givenUpWeight = weights.get('givenUp', 0.0)

    g_weights.maxDelayWeight = weights.get('maxDelay', 0.0)
    g_weights.avgDelayWeight = weights.get('avgDelay', 0.0)
    g_weights.avgJitterWeight = weights.get('avgJitter', 0.0)
    g_weights.avgTcpSenderSrttWeight = weights.get('avgTcpSenderSrtt', 0.0)

    g_weights.dropRatioWeight = weights.get('dropRatio', 0.0)
    g_weights.inFlightRatioWeight = weights.get('inFlightRatio', 0.0)
    g_weights.maxLinkUtilizationWeight = weights.get('maxLinkUtilization', 0.0)
    g_weights.avgLinkUtilizationWeight = weights.get('avgLinkUtilization', 0.0)
    g_weights.maxTxQueueRelativeLoadWeight = weights.get('maxTxQueueRelativeLoad', 0.0)
    g_weights.avgTxQueueRelativeLoadWeight = weights.get('avgTxQueueRelativeLoad', 0.0)

    g_weights.transmittedWeight = weights.get('transmitted', 0.0)
    g_weights.macTxDroppedWeight = weights.get('macTxDropped', 0.0)
    g_weights.phyRxDroppedWeight = weights.get('phyRxDropped', 0.0)

    return g_weights


# ======================================================================================================================


def observation_ndarray_to_pyg(global_data: np.ndarray,
                               node_data: np.ndarray,
                               edge_data: np.ndarray,
                               V, E,
                               node_dist_provider: ShortestPathCalculator) -> (Observation, np.ndarray, int):
    """
    Converts given observation data (global, node, edge) into a list of observation graphs, each from the perspective
    of a specific observer, and puts them into a single Observation object. Additionally, calculates communication
    delays and central node from the birdseye observation graph.

    Args:
        global_data: structured array with global observations
        node_data: structured array with node observations
        edge_data: structured array with edge observations
        V: number of nodes
        E: number of edges
        node_dist_provider: a shortest-path calculator for calculating communication delays and central node
    """

    # for global/node/edge observations we can assume that the observed entries are sorted by observerId,
    # and thus share the same observerId per corresponding slice.
    all_global_observer_ids = global_data['observerId']  # shape: (num_observers,)
    all_global_feat = torch.from_numpy(
        np.delete(structured_to_unstructured(global_data, dtype=np.double), global_meta_idx, axis=1)
        )  # shape: (num_observers, num_features)
    # in the realm of PyG, node IDs are contiguous,
    # so we call our routing node IDs "agent IDs" from here to avoid confusion
    all_agent_ids = torch.split(
        torch.from_numpy(
            np.ascontiguousarray(node_data['nodeId'], dtype=np.int64)
        ), V, dim=0)
    all_node_feat = torch.split(
        torch.from_numpy(
            np.delete(structured_to_unstructured(node_data, dtype=np.double), node_meta_idx, axis=1)
        ), V, dim=0)  # tuple with num_observers elements, each of shape (num_nodes, num_features)

    all_edge_is_link_up = np.ascontiguousarray(edge_data['isLinkUp'], dtype=bool).reshape(-1, E)  # shape: (num_observers, num_edges)
    all_edge_index = torch.split(
        torch.from_numpy(
            np.stack((np.ascontiguousarray(edge_data['src'], dtype=np.int64),
                      np.ascontiguousarray(edge_data['dst'], dtype=np.int64)), axis=0)
        ), E, dim=1)  # tuple with num_observers elements, each of shape (2, num_edges)
    all_edge_feat = torch.split(
        torch.from_numpy(
            np.delete(structured_to_unstructured(edge_data, dtype=np.double), edge_meta_idx, axis=1)
        ), E, dim=0)  # tuple with num_observers elements, each of shape (num_edges, num_features)

    # iterate over the zip of global, node and edge observations to convert to observation graphs
    observation_graphs = []
    comm_delays, central_node = None, None
    for i, (observer_id, global_feat, agent_ids, node_feat, edge_is_link_up, edge_index, edge_feat) \
            in enumerate(zip(all_global_observer_ids, all_global_feat, all_agent_ids, all_node_feat,
                             all_edge_is_link_up, all_edge_index, all_edge_feat)):

        # If the number of read entries is less that the maximum number used to preallocate buffer memory,
        #  buffer entries beyond the received ones (V nodes + 1 global observer) will be garbage -> skip them
        if i > V+1:
            break

        # we use agent_ids (sorted) as a mapping from the actual 'agent' IDs -> inner, contiguous node IDs,
        # as required by PyTorch Geometric, to adjust edge_index accordingly
        outer_src, outer_dst = edge_index[:, edge_is_link_up]  # filter out links that are down
        inner_src = torch.searchsorted(agent_ids, outer_src)
        inner_dst = torch.searchsorted(agent_ids, outer_dst)
        inner_edge_index = torch.stack([inner_src, inner_dst], dim=0)  # shape: (2, num_edges)

        obs_graph = ObservationGraph(x=node_feat,  # shape: (num_nodes, num_node_features)
                                     edge_index=inner_edge_index,
                                     edge_attr=edge_feat[edge_is_link_up],  # shape: (num_edges with isLinkUp=True, num_edge_features)
                                     u=global_feat.unsqueeze(0),  # shape: (1, num_global_features)
                                     agent_ids=agent_ids,  # shape: (num_nodes,). Necessary for agent mapping in learning
                                     observer_ids=torch.tensor([observer_id], dtype=torch.long)  # shape: (1,)
                                     )

        # validity check: we currently only support observation graphs where each node has at least one outgoing edge
        if not obs_graph.each_node_has_outgoing_edges():
            raise ValueError(f"Observation graph with observerId {observer_id} has nodes without outgoing edges! "
                             f"Such graphs are currently not supported.")

        # calculate communication delays and central node from birdseye observation graph
        if observer_id == BIRDSEYE_OBSERVER_ID:
            comm_delays, central_node = node_dist_provider.get_node_distances_and_central_node(obs_graph)
        observation_graphs.append(obs_graph)

    # comm_delays and central node should have been calculated
    if comm_delays is None or central_node is None:
        raise ValueError(f"Observation graph with observerId {BIRDSEYE_OBSERVER_ID} not found! "
                         f"Didn't calculate communication delays and central node.")

    obs = Observation.from_obs_graph_list(obs_graph_list=observation_graphs,
                                          central_node=torch.tensor(central_node, dtype=torch.long))
    return obs, comm_delays, central_node


def reward_ndarray_to_pyg(reward_data: np.ndarray) -> torch.Tensor:
    """
C   onverts given reward data (structured numpy array) into a flattened PyTorch tensor.
    """
    all_rewards = torch.from_numpy(
        np.delete(structured_to_unstructured(reward_data, dtype=np.double), reward_meta_idx, axis=1)
    ).flatten()  # shape: (num_observers,)
    return all_rewards
