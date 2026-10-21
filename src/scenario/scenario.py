from typing import List, Dict, Any

import numpy as np
import networkx as nx

from .events import Event, LinkFailure, FlowArrival, EventsConfig
from .topology import TopologyConfig
from .config import BaseConfig
from src.utils.canonical_bytes import nx_graph_to_canonical_bytes, canonical_bytes, canonical_hash


class ScenarioConfig(BaseConfig):
    """
    Configuration for a network scenario. Contains the configuration for the topology and the events.
    """
    seed: int
    scenario_length: int  # number of timesteps in the scenario
    ms_per_step: float  # duration per simulation step, also scales traffic volume
    rng_uniform_deviation: float  # sample uniform from [1-deviation, 1+deviation]
    topology: TopologyConfig
    events: EventsConfig


class NetworkScenario:
    """
    A network scenario consists of a network topology and a sequence of events that occur in the network.
    These events can be traffic flows or link failures.
    """

    def __init__(self,
                 network,
                 events,
                 ms_per_step,
                 buffer_size_bdp,
                 prob_tcp,
                 use_tcp_sack,
                 onehot_id,
                 visualize):
        self.network: nx.Graph = network
        self.events: List[List[Event]] = events

        self.length = len(self.events)
        self.num_events_per_step = [len(events_t) for events_t in self.events]

        self.ms_per_step = ms_per_step
        self.buffer_size_bdp = buffer_size_bdp
        self.prob_tcp = prob_tcp
        self.use_tcp_sack = use_tcp_sack
        self.onehot_id: np.ndarray = onehot_id
        self.visualize = visualize

    def get_event_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about the events in the scenario.
        """
        num_events, num_traffic_flow_arrivals, num_link_failures = 0, 0, 0
        flow_sizes = []

        for events_t in self.events:
            for e in events_t:
                num_events += 1
                if isinstance(e, FlowArrival):
                    num_traffic_flow_arrivals += 1
                    flow_sizes.append(e.amount)
                elif isinstance(e, LinkFailure):
                    num_link_failures += 1

        ds_p10, ds_25, ds_50, ds_75, ds_90, ds_max = np.percentile(flow_sizes, [10, 25, 50, 75, 90, 100])

        event_stats = {
            "num_events": num_events,
            "traffic": {
                "num_traffic_flows": num_traffic_flow_arrivals,
                "num_link_failures": num_link_failures,
                "flow_sizes_avg": np.average(flow_sizes),
                "flow_sizes_p10": ds_p10,
                "flow_sizes_p25": ds_25,
                "flow_sizes_p75": ds_75,
                "flow_sizes_p90": ds_90,
                "flow_size_max": ds_max,
            }
        }
        return event_stats

    def get_topology_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about the topology in the scenario.
        """
        topology_stats = {
            "graph": {
                "num_nodes": self.network.number_of_nodes(),
                "num_edges": self.network.number_of_edges(),
            },
        }
        return topology_stats

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns statistics about the scenario.
        """
        return {"topology": self.get_topology_stats(), "events": self.get_event_stats()}

    def get_hash_strs(self) -> (str, str, str):
        """
        Returns a hashable string representation of the scenario (network, events, and their placement in time).
        """
        topology_bytes = nx_graph_to_canonical_bytes(self.network)
        event_parts = [e.to_bytes() for events_t in self.events for e in events_t]
        num_events_bytes = canonical_bytes(self.num_events_per_step)

        scenario_hash = canonical_hash([topology_bytes] + event_parts + [num_events_bytes])
        topology_hash = canonical_hash([topology_bytes])
        events_hash = canonical_hash(event_parts + [num_events_bytes])
        return scenario_hash, topology_hash, events_hash
