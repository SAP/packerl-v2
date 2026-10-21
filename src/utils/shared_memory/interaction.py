"""
This module provides functions to place data into and read data from shared memory.
"""
from copy import deepcopy
import ctypes
from typing import List, Any, Dict

import networkx as nx
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured
from py_interface import Ns3AIRL as ns3Sim
from torch import Tensor

from src.scenario.events import Event as PSEvent
from src.utils.actions import PackerlEnvAction
from src.utils.logging import LOG_CONSOLE_FUNCTION, LOG_CONSOLE_LOGIC, LOG_CONSOLE_DEBUG
from src.utils.observations import Observation
from src.utils.types import TrafficMatrix, Reward
from src.utils.shared_memory.structs import (MAX_ELEMS,
                                             CommunicationDelay,
                                             comm_delay_dtype,
                                             ObservedGlobal,
                                             observed_global_dtype,
                                             ObservedNode,
                                             observed_node_dtype,
                                             ObservedEdge,
                                             observed_edge_dtype,
                                             ObservedReward,
                                             observed_reward_dtype,
                                             ObservedTrafficEntry,
                                             RoutingActionComponent,
                                             RoutingAgentDelay,
                                             )
from src.utils.time import TimeoutContext
from src.utils.shared_memory.conversion import (topology_edge_to_ctype,
                                                event_to_ctype,
                                                actions_to_bytes,
                                                agent_delays_to_bytes,
                                                global_reward_weights_to_ctype,
                                                observation_ndarray_to_pyg, reward_ndarray_to_pyg
                                                )


class SharedMemUnavailableException(Exception):
    """
    A Custom exception thrown when the shared memory object provided via a context manager is None.
    This indicates that the shared memory is unavailable, e.g. due to a segfault on the C++ side.
    """
    def __init__(self):
        self.message = "Couldn't access shared memory from py side -> terminating..."
        super().__init__(self.message)


def check_shm(sim):
    """
    Check whether ns3 is ready to interact via shared memory
    """
    LOG_CONSOLE_FUNCTION()
    sim_ready = False  # sim is ready if done flag is set to False
    while not sim_ready:
        with sim as data:

            # ns3 not ready or some error
            if not data:
                raise SharedMemUnavailableException()
            sim_ready = data.env.simReady

    LOG_CONSOLE_FUNCTION()


# === PLACING =========================================================================================================


def place_done(sim: ns3Sim, done: bool, ns3_proc) -> None:
    """
    Sets the done flag in shared memory to signify the sim that the current episode is over
    """
    LOG_CONSOLE_FUNCTION()
    with TimeoutContext(sim, ns3_proc) as data:

        # ns3 not ready or some error
        if not data:
            raise SharedMemUnavailableException()
        if data.env.doneAvailable:
            LOG_CONSOLE_DEBUG("ns3 has not yet consumed previously sent done flag")
            raise TimeoutContext.Break  # release shared memory
        data.env.done = done
        data.env.doneAvailable = True
    LOG_CONSOLE_FUNCTION()


def place_network_graph(sim: ns3Sim, network_graph: nx.DiGraph, ns3_proc) -> None:
    """
    Places the given network graph into the shared memory so that
    the simulation can read and install it.
    """
    LOG_CONSOLE_FUNCTION()
    remaining_edges = list(network_graph.edges(data=True))
    while len(remaining_edges) > 0:
        with TimeoutContext(sim, ns3_proc) as data:

            # ns3 not ready or some error
            if not data:
                raise SharedMemUnavailableException()
            if data.env.topologyAvailable:
                LOG_CONSOLE_LOGIC("ns3 has not yet consumed past topology")
                raise TimeoutContext.Break  # release shared memory

            # if the remaining edges are less than MAX_ELEMS, send all of them (and general info)
            if len(remaining_edges) <= MAX_ELEMS:
                for i, (fst, snd, edge_attrs) in enumerate(remaining_edges):
                    LOG_CONSOLE_DEBUG(f"sending edge {fst}->{snd}")
                    data.env.topologyEdges[i] = topology_edge_to_ctype(fst, snd, edge_attrs)
                data.env.numTopologyEdgesAvailable = len(remaining_edges)
                data.env.allTopologyEdgesSent = True
                remaining_edges = []

            # if the remaining edges are more than MAX_ELEMS, send MAX_ELEMS of them
            else:
                edges_to_be_sent = remaining_edges[:MAX_ELEMS]
                for i, (fst, snd, edge_attrs) in enumerate(edges_to_be_sent):
                    LOG_CONSOLE_DEBUG(f"sending edge {fst}->{snd}")
                    data.env.topologyEdges[i] = topology_edge_to_ctype(fst, snd, edge_attrs)
                data.env.numTopologyEdgesAvailable = len(edges_to_be_sent)
                data.env.allTopologyEdgesSent = False
                remaining_edges = remaining_edges[MAX_ELEMS:]

            data.env.numTopologyNodes = network_graph.number_of_nodes()
            data.env.topologyAvailable = True  # signals ns3 that the topology is ready to be consumed

    # wait until ns3 fully consumes the data, then yield shm control to ns3 (by simply accessing the shm)
    data_consumed = False
    while not data_consumed:
        with TimeoutContext(sim, ns3_proc) as data:
            data_consumed = not data.env.topologyAvailable
    LOG_CONSOLE_FUNCTION()


def place_global_reward_weights(sim: ns3Sim, global_reward_weights: Dict[str, float], ns3_proc) -> None:
    """
    Places the given global reward weights into the shared memory so that the simulation can read and use them.
    """
    LOG_CONSOLE_FUNCTION()
    with TimeoutContext(sim, ns3_proc) as data:

        # ns3 not ready or some error
        if not data:
            raise SharedMemUnavailableException()
        if data.env.globalRewardWeightsAvailable:
            LOG_CONSOLE_DEBUG("ns3 has not yet consumed previously sent global reward weights")
            raise TimeoutContext.Break  # release shared memory

        data.env.globalRewardWeights = global_reward_weights_to_ctype(global_reward_weights)
        data.env.globalRewardWeightsAvailable = True

    # wait until ns3 fully consumes the data, then yield shm control to ns3 (by simply accessing the shm)
    data_consumed = False
    while not data_consumed:
        with TimeoutContext(sim, ns3_proc) as data:
            data_consumed = not data.env.globalRewardWeightsAvailable
    LOG_CONSOLE_FUNCTION()


def place_actions(sim: ns3Sim, actions_with_delays: (PackerlEnvAction, Tensor), ns3_proc) -> None:
    """
    Places the given actions into the shared memory so that the simulation can read and execute them.
    """
    LOG_CONSOLE_FUNCTION()

    actions, delays_per_agent = actions_with_delays

    # convert actions to bytes
    action_components_bytes, total_action_components = actions_to_bytes(actions)
    action_components_buffer = (ctypes.c_ubyte * total_action_components * ctypes.sizeof(RoutingActionComponent))()
    ctypes.memmove(ctypes.addressof(action_components_buffer), action_components_bytes, len(action_components_bytes))

    # convert action delays to bytes
    agent_delays_bytes, total_action_delays = agent_delays_to_bytes(actions.agent_ids, delays_per_agent)
    agent_delays_buffer = (ctypes.c_ubyte * total_action_delays * ctypes.sizeof(RoutingAgentDelay))()
    ctypes.memmove(ctypes.addressof(agent_delays_buffer), agent_delays_bytes, len(agent_delays_bytes))

    ac_idx = 0
    all_actions_sent = False
    dl_idx = 0
    all_agent_delays_sent = False
    while not (all_actions_sent and all_agent_delays_sent):
        with TimeoutContext(sim, ns3_proc) as data:

            # ns3 not ready or some error
            if not data:
                raise SharedMemUnavailableException()

            if data.act.actionsAvailable or data.act.routingAgentDelaysAvailable:
                LOG_CONSOLE_LOGIC("ns3 has not yet consumed previously sent action data")
                raise TimeoutContext.Break  # release shared memory

            # 1. routing actions
            remaining_action_components = total_action_components - ac_idx
            if remaining_action_components <= MAX_ELEMS:
                all_actions_sendable = True
                num_upcoming_actions = remaining_action_components
            else:
                all_actions_sendable = False
                num_upcoming_actions = MAX_ELEMS
            # place next 'num_upcoming_actions' entries from 'action_components_buffer', starting from 'ac_idx',
            # into the shared memory at position 'data.act.routingActions', and update 'ac_idx'.
            ctypes.memmove(data.act.routingActions,
                           ctypes.addressof(action_components_buffer) + ac_idx * ctypes.sizeof(RoutingActionComponent),
                           num_upcoming_actions * ctypes.sizeof(RoutingActionComponent))
            ac_idx += num_upcoming_actions

            # 2. agent delays
            remaining_agent_delays = total_action_delays - dl_idx
            if remaining_agent_delays <= MAX_ELEMS:
                all_agent_delays_sendable = True
                num_upcoming_agent_delays = remaining_agent_delays
            else:
                all_agent_delays_sendable = False
                num_upcoming_agent_delays = MAX_ELEMS
            # place next 'num_upcoming_agent_delays' entries from 'agent_delays_buffer', starting from 'dl_idx',
            # into the shared memory at position 'data.act.routingAgentDelays', and update 'dl_idx'.
            ctypes.memmove(data.act.routingAgentDelays,
                           ctypes.addressof(agent_delays_buffer) + dl_idx * ctypes.sizeof(RoutingAgentDelay),
                           num_upcoming_agent_delays * ctypes.sizeof(RoutingAgentDelay))
            dl_idx += num_upcoming_agent_delays

            # bookkeeping and signaling
            all_actions_sent = all_actions_sendable
            data.act.numRoutingActionsAvailable = num_upcoming_actions
            data.act.allActionsSent = all_actions_sent
            data.act.actionsAvailable = True  # signals ns3 that the actions are ready to be consumed
            all_agent_delays_sent = all_agent_delays_sendable
            data.act.numRoutingAgentDelaysAvailable = num_upcoming_agent_delays
            data.act.allRoutingAgentDelaysSent = all_agent_delays_sent
            data.act.routingAgentDelaysAvailable = True  # signals ns3 that the agent delays are ready to be consumed

    # wait until ns3 fully consumes the data, then yield shm control to ns3 (by simply accessing the shm)
    data_consumed = False
    while not data_consumed:
        with TimeoutContext(sim, ns3_proc) as data:
            data_consumed = not (data.act.actionsAvailable or data.act.routingAgentDelaysAvailable)
    LOG_CONSOLE_FUNCTION()


def place_events(sim: ns3Sim, events: List[PSEvent], ns3_proc) -> None:
    """
    Places the given events into the shared memory so that the simulation can read and execute them.
    """
    LOG_CONSOLE_FUNCTION()
    remaining_events = deepcopy(events)
    all_upcoming_events_sent = False

    while not all_upcoming_events_sent:
        with TimeoutContext(sim, ns3_proc) as data:

            # ns3 not ready or some error
            if not data:
                raise SharedMemUnavailableException()
            if data.env.upcomingEventsAvailable:
                LOG_CONSOLE_LOGIC("ns3 has not yet consumed previously sent upcoming events")
                raise TimeoutContext.Break  # release shared memory

            # check whether we can send all events this time
            if len(remaining_events) <= MAX_ELEMS:
                all_events_sendable = True
                events_to_send = remaining_events
                new_remaining_events = []
            else:
                all_events_sendable = False
                events_to_send = remaining_events[:MAX_ELEMS]
                new_remaining_events = remaining_events[MAX_ELEMS:]

            # place the events
            for i, event in enumerate(events_to_send):
                LOG_CONSOLE_DEBUG(f"sending event '{event}'")
                c_event_type_id, c_event = event_to_ctype(event)
                c_event_field = "trafficFlowArrival" if c_event_type_id == 0 else "linkFailure"
                setattr(data.env.upcomingEvents[i], c_event_field, c_event)  # set the event depending on type
                data.env.upcomingEventTypes[i] = c_event_type_id

            # bookkeeping and signaling
            all_upcoming_events_sent = all_events_sendable
            data.env.numUpcomingEventsAvailable = len(events_to_send)
            data.env.allUpcomingEventsSent = all_upcoming_events_sent
            data.env.upcomingEventsAvailable = True  # signals ns3 that the events are ready to be consumed
            remaining_events = new_remaining_events
            LOG_CONSOLE_LOGIC(f"Remaining events: {len(remaining_events)}")

    # wait until ns3 fully consumes the data, then yield shm control to ns3 (by simply accessing the shm)
    data_consumed = False
    while not data_consumed:
        with TimeoutContext(sim, ns3_proc) as data:
            data_consumed = not data.env.upcomingEventsAvailable
    LOG_CONSOLE_FUNCTION()


def place_communication_delays(sim: ns3Sim, comm_delays_and_central_node: (np.ndarray, int), ns3_proc) -> None:
    """
    Places the given communication delays into the shared memory so that the simulation can read and use them.
    """
    LOG_CONSOLE_FUNCTION()
    comm_delays, central_node = comm_delays_and_central_node
    total_comm_delays = comm_delays.shape[0]
    comm_delays_bytes = unstructured_to_structured(comm_delays, dtype=comm_delay_dtype).tobytes()
    comm_delays_buffer = (ctypes.c_ubyte * len(comm_delays_bytes) * ctypes.sizeof(CommunicationDelay))()
    ctypes.memmove(ctypes.addressof(comm_delays_buffer), comm_delays_bytes, len(comm_delays_bytes))

    comm_delay_idx = 0
    all_comm_delays_sent = False
    while not all_comm_delays_sent:
        with TimeoutContext(sim, ns3_proc) as data:

            # ns3 not ready or some error
            if not data:
                raise SharedMemUnavailableException()
            if data.env.communicationDelaysAvailable:
                LOG_CONSOLE_DEBUG("ns3 has not yet consumed previously sent communication delays")
                raise TimeoutContext.Break  # release shared memory

            # check whether we can send all communication delays this time
            remaining_comm_delays = total_comm_delays - comm_delay_idx
            if remaining_comm_delays <= MAX_ELEMS:
                all_comm_delays_sendable = True
                num_upcoming_delays = remaining_comm_delays
            else:
                all_comm_delays_sendable = False
                num_upcoming_delays = MAX_ELEMS

            # place the next 'num_upcoming_delays' entries from 'comm_delays_buffer', starting from 'comm_delay_idx',
            # into the shared memory at position 'data.env.communicationDelays', and update 'comm_delay_idx'.
            ctypes.memmove(data.env.communicationDelays,
                           ctypes.addressof(comm_delays_buffer) + comm_delay_idx * ctypes.sizeof(CommunicationDelay),
                           num_upcoming_delays * ctypes.sizeof(CommunicationDelay))
            comm_delay_idx += num_upcoming_delays

            # bookkeeping and signaling
            all_comm_delays_sent = all_comm_delays_sendable
            data.env.numCommunicationDelaysAvailable = num_upcoming_delays
            data.env.allCommunicationDelaysSent = all_comm_delays_sent
            data.env.centralNode = central_node
            data.env.communicationDelaysAvailable = True

    # wait until ns3 fully consumes the data, then yield shm control to ns3 (by simply accessing the shm)
    data_consumed = False
    while not data_consumed:
        with TimeoutContext(sim, ns3_proc) as data:
            data_consumed = not data.env.communicationDelaysAvailable
    LOG_CONSOLE_FUNCTION()


def place_shm(sim: ns3Sim, mode: str, data: Any, ns3_proc):
    """
    Places the given data into the shared memory according to the given mode.
    """
    if mode == "done":
        place_done(sim, data, ns3_proc)
    elif mode == "network_graph":
        place_network_graph(sim, data, ns3_proc)
    elif mode == "global_reward_weights":
        place_global_reward_weights(sim, data, ns3_proc)
    elif mode == "actions":
        place_actions(sim, data, ns3_proc)
    elif mode == "events":
        place_events(sim, data, ns3_proc)
    elif mode == "communication_delays":
        place_communication_delays(sim, data, ns3_proc)
    else:
        raise ValueError(f"Unknown shm placement mode: {mode}")


# === READING =========================================================================================================


def read_shm(max_V: int, max_E: int, sim, node_dist_provider, ns3_proc) \
        -> (Observation, TrafficMatrix, TrafficMatrix, Reward, float, float, np.ndarray, int):
    """
    Reads the observation components (observation graphs, traffic matrices etc.) from shared memory and converts them
    to the corresponding Python objects.
    """
    LOG_CONSOLE_FUNCTION()

    # Pre-allocate byte buffers for the observed components, using upper bounds defined by the provided max_V and max_E
    MAX_GLOBAL_ENTRIES = max_V + 1
    MAX_NODE_ENTRIES = (max_V + 1) * max_V
    MAX_EDGE_ENTRIES = (max_V + 1) * max_E
    MAX_REWARD_ENTRIES = max_V + 1
    MAX_TRAFFIC_ENTRIES = 2 * max_V * max_V
    global_buffer = (ctypes.c_ubyte * MAX_GLOBAL_ENTRIES * ctypes.sizeof(ObservedGlobal))()
    node_buffer = (ctypes.c_ubyte * MAX_NODE_ENTRIES * ctypes.sizeof(ObservedNode))()
    edge_buffer = (ctypes.c_ubyte * MAX_EDGE_ENTRIES * ctypes.sizeof(ObservedEdge))()
    reward_buffer = (ctypes.c_ubyte * MAX_REWARD_ENTRIES * ctypes.sizeof(ObservedReward))()
    traffic_buffer = (ctypes.c_ubyte * MAX_TRAFFIC_ENTRIES * ctypes.sizeof(ObservedTrafficEntry))()

    # Indices to keep track of the current position in the buffers. Also tells the number of read entries at the end.
    global_idx = 0
    node_idx = 0
    edge_idx = 0
    reward_idx = 0
    traffic_idx = 0

    monitoring_time = None
    elapsed_time = None
    actual_V = None
    actual_E = None

    # shm interaction bookkeeping
    observed_info_received = False
    observed_globals_received = False
    observed_nodes_received = False
    observed_edges_received = False
    observed_rewards_received = False
    observed_traffic_entries_received = False
    observation_received = (observed_info_received
                            and observed_globals_received
                            and observed_nodes_received
                            and observed_edges_received
                            and observed_rewards_received
                            and observed_traffic_entries_received)

    while not observation_received:
        # extended timeout because ns-3 does the simulation step in the meantime
        with TimeoutContext(sim, ns3_proc, shm_access_timeout=300) as data:

            # ns3 not ready or some error
            if not data:
                raise SharedMemUnavailableException()
            if not data.env.observationAvailable:
                LOG_CONSOLE_LOGIC("ns3 has not yet sent observation")
                raise TimeoutContext.Break  # release shared memory

            # observed info (only read once)
            if not observed_info_received:
                monitoring_time = data.env.observedInfo.monitoringTime
                elapsed_time = data.env.observedInfo.elapsedTime
                actual_V = data.env.observedInfo.numNodes
                actual_E = data.env.observedInfo.numEdges
                observed_info_received = True

            # observed globals
            if not observed_globals_received:
                num_available_globals = data.env.numAvailableObservedGlobals
                ctypes.memmove(ctypes.addressof(global_buffer) + global_idx * ctypes.sizeof(ObservedGlobal),
                               data.env.observedGlobals,
                               num_available_globals * ctypes.sizeof(ObservedGlobal))
                global_idx += num_available_globals
                observed_globals_received = data.env.allObservedGlobalsSent

            # observed nodes
            if not observed_nodes_received:
                num_available_nodes = data.env.numAvailableObservedNodes
                ctypes.memmove(ctypes.addressof(node_buffer) + node_idx * ctypes.sizeof(ObservedNode),
                               data.env.observedNodes,
                               num_available_nodes * ctypes.sizeof(ObservedNode))
                node_idx += num_available_nodes
                observed_nodes_received = data.env.allObservedNodesSent

            # observed edges
            if not observed_edges_received:
                num_available_edges = data.env.numAvailableObservedEdges
                ctypes.memmove(ctypes.addressof(edge_buffer) + edge_idx * ctypes.sizeof(ObservedEdge),
                               data.env.observedEdges,
                               num_available_edges * ctypes.sizeof(ObservedEdge))
                edge_idx += num_available_edges
                observed_edges_received = data.env.allObservedEdgesSent

            # observed rewards
            if not observed_rewards_received:
                num_available_rewards = data.env.numAvailableObservedRewards
                ctypes.memmove(ctypes.addressof(reward_buffer) + reward_idx * ctypes.sizeof(ObservedReward),
                               data.env.observedRewards,
                               num_available_rewards * ctypes.sizeof(ObservedReward))
                reward_idx += num_available_rewards
                observed_rewards_received = data.env.allObservedRewardsSent

            # observed matrix entries
            if not observed_traffic_entries_received:
                num_available_traffic_entries = data.env.numAvailableObservedTrafficEntries
                ctypes.memmove(ctypes.addressof(traffic_buffer) + traffic_idx * ctypes.sizeof(ObservedTrafficEntry),
                               data.env.observedTrafficEntries,
                               num_available_traffic_entries * ctypes.sizeof(ObservedTrafficEntry))
                traffic_idx += num_available_traffic_entries
                observed_traffic_entries_received = data.env.allObservedTrafficEntriesSent


            # mark observation as consumed, next observation should only be read once the next observation is provided
            data.env.observationAvailable = False
            observation_received = (observed_info_received
                                    and observed_globals_received
                                    and observed_nodes_received
                                    and observed_edges_received
                                    and observed_rewards_received
                                    and observed_traffic_entries_received)

    # observation (graphs per observer)
    observation, comm_delays, central_node = observation_ndarray_to_pyg(
        np.frombuffer(global_buffer, dtype=observed_global_dtype),
        np.frombuffer(node_buffer, dtype=observed_node_dtype),
        np.frombuffer(edge_buffer, dtype=observed_edge_dtype),
        actual_V, actual_E,
        node_dist_provider
    )

    # traffic matrices are sent for global and central node observation
    traffic_data = np.frombuffer(traffic_buffer, dtype=np.uint64)[:traffic_idx]
    global_traffic_matrix, central_node_traffic_matrix \
        = np.split(traffic_data.reshape((2*actual_V, actual_V)), [actual_V], axis=0)

    # reward (one scalar per observer, packed into a tuple of two tensors)
    reward_tensor = reward_ndarray_to_pyg(np.frombuffer(reward_buffer, dtype=observed_reward_dtype)[:reward_idx])
    assert reward_tensor.shape == observation.observer_ids.shape, "Reward tensor shape mismatch"
    rewards: Reward = reward_tensor, observation.observer_ids

    LOG_CONSOLE_FUNCTION(end=True)
    return (observation,
            global_traffic_matrix,
            central_node_traffic_matrix,
            rewards,
            monitoring_time,
            elapsed_time,
            comm_delays,
            central_node)
