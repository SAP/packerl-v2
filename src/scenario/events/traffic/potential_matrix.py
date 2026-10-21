from typing import List

import networkx as nx
import numpy as np

from src.scenario.config import BaseConfig, SingleOrChoice
from src.scenario.events.event import Event
from src.scenario.events.traffic.flow_arrival import FlowArrival
from src.utils.logging import LOG_CONSOLE_FUNCTION, LOG_CONSOLE_DEBUG

INTERARRIVAL_SAMPLE_SIZE = 10_000  # number of samples to generate at a time for the interarrival time distribution


class PotentialMatrixConfig(BaseConfig):
    """
    Configuration for the potential matrix-based traffic generation.
    """
    traffic_scaling: SingleOrChoice[float]
    flow_size_tradeoff: SingleOrChoice[float]  # from 0.0 (more but smaller flows) to 1 (fewer but larger flows)
    max_flow_interarrival_time: int  # maximum time between two flows in seconds
    flow_size_distribution_scale: float  # scale parameter of the flow size distribution
    flow_size_distribution_shape_base: float  # shape parameter of the flow size distribution
    flow_interarrival_distribution_shape: float  # shape parameter of the flow interarrival time distribution


def create_potential_matrix(G: nx.Graph,
                            rng: np.random.Generator,
                            scenario_cfg: dict,
                            total_time,
                            check=False) -> np.ndarray:
    """
    Generate a potential matrix based on the traffic potentials of the nodes in the graph.
    """
    node_traffic_potentials = list(nx.get_node_attributes(G, 'traffic_potential').values())
    node_traffic_potentials = np.array(node_traffic_potentials)[:, np.newaxis]
    if check:
        assert np.all(node_traffic_potentials >= 0)
    potential_matrix = np.matmul(node_traffic_potentials, node_traffic_potentials.T)  # create potential matrix
    potential_matrix = potential_matrix * rng.uniform(low=1 - scenario_cfg['rng_uniform_deviation'],
                          high=1 + scenario_cfg['rng_uniform_deviation'],
                          size=potential_matrix.shape)  # random perturbation
    np.fill_diagonal(potential_matrix, 0)
    potential_matrix = (total_time * potential_matrix) / np.sum(potential_matrix)  # normalize to sum=total_time
    if check:
        assert np.all(potential_matrix >= 0)

    return potential_matrix


def get_datarates(rng: np.random.Generator,
                  flow_sizes: np.array,
                  flow_is_tcp: np.array) -> np.array:
    """
    Generate data rates for the traffic flows based on the flow sizes and the flow types.
    """
    data_rates = np.zeros_like(flow_sizes)

    # Apply conditions, excluding where flow_is_tcp is True (because these will adjust their data rates themselves)
    small_udp_flow = ~flow_is_tcp & (flow_sizes < 1e5)
    large_udp_flow = ~flow_is_tcp & (flow_sizes >= 1e5)

    # Generate data rates based on conditions
    data_rates[small_udp_flow] = 1e9 * np.ones(np.sum(small_udp_flow))
    data_rates[large_udp_flow] = rng.uniform(1e6, 5e6, size=np.sum(large_udp_flow))

    return data_rates.astype(int)


def generate_flows_from_potential_matrix(G: nx.Graph,
                                         events: List[List[Event]],
                                         scenario_cfg: dict,
                                         rng: np.random.Generator,
                                         check=False):
    """
    Generate traffic flow events based on the configuration provided in the scenario configuration.
    """
    LOG_CONSOLE_FUNCTION()

    # preparation
    traffic_cfg = scenario_cfg['events']['traffic']
    pm_config = traffic_cfg['mode_configs']['potential_matrix']
    traffic_scaling = pm_config['traffic_scaling']
    prob_tcp = traffic_cfg['prob_tcp']
    flow_size_tradeoff = pm_config["flow_size_tradeoff"]
    ms_per_step = scenario_cfg['ms_per_step']
    total_time = scenario_cfg['scenario_length'] * ms_per_step
    max_interarrival_time = min(pm_config['max_flow_interarrival_time'], total_time / 4)  # in ms

    # log-logistic interarrival distribution:
    #  - Uses uniform samples and a quantile function, then caps interarrival time
    #  - scale is adjusted based on flow_size_tradeoff and traffic scalings of scenario config and current graph
    did_shape = pm_config["flow_interarrival_distribution_shape"]
    did_scale = (flow_size_tradeoff + 0.2) / (traffic_scaling * G.graph['traffic_scaling'])

    def sample_interarrivals(size):
        uniform_samples = rng.uniform(0, 1, size)
        log_logistic_samples = did_scale * ((uniform_samples / (1 - uniform_samples)) ** (1 / did_shape))
        return np.minimum(log_logistic_samples, max_interarrival_time)

    # pareto flow size distribution: Samples from Lomax distribution (called 'pareto' in np) and post-processes
    dsd_shape = pm_config['flow_size_distribution_shape_base'] + np.log(np.power(0.1 + flow_size_tradeoff, -1 / 37))
    dsd_scale = pm_config['flow_size_distribution_scale']

    def sample_flow_sizes(size):
        lomax_samples = rng.pareto(dsd_shape, size)
        pareto_samples = dsd_scale * (lomax_samples + 1)
        return np.minimum(pareto_samples, traffic_cfg['max_flow_size']).astype(int)

    # create potential matrix based on the traffic potentials of the nodes
    potential_matrix = create_potential_matrix(G, rng, scenario_cfg, total_time, check=check)

    # get interarrival times (for all src-dst pairs at once, we'll assign them to specific src-dst pairs later)
    interarrival_times = sample_interarrivals(INTERARRIVAL_SAMPLE_SIZE)
    cumsum_time = np.sum(interarrival_times)
    while cumsum_time < np.sum(potential_matrix):
        new_samples = sample_interarrivals(INTERARRIVAL_SAMPLE_SIZE)
        interarrival_times = np.append(interarrival_times, new_samples)
        cumsum_time += np.sum(new_samples)
    if check:
        assert np.all(interarrival_times >= 0)

    LOG_CONSOLE_DEBUG("Sampled interarrivals")

    # get number of flows and splitting indices, depending on the potential matrix
    ia_time_cumsum = np.cumsum(interarrival_times)  # [0.0 ... smth. >= (np.sum(pm) * total_time)]
    pm_flat = potential_matrix.flatten()
    cum_flow_split_idx = np.searchsorted(ia_time_cumsum, np.cumsum(pm_flat), side='right')
    num_flows_per_pm_entry = np.diff(cum_flow_split_idx, prepend=0)
    if check:
        assert np.all(num_flows_per_pm_entry >= 0)
    num_flows = np.sum(num_flows_per_pm_entry)

    LOG_CONSOLE_DEBUG(f"Got number of flows: {num_flows}")

    # remove extra samples and overflowing split indices and thresholds
    # (this happens whenever we generated exactly the number of flows needed)
    ia_time_cumsum = ia_time_cumsum[:num_flows]
    pm_flat_pruned = pm_flat[cum_flow_split_idx < num_flows]
    cum_flow_split_idx = cum_flow_split_idx[cum_flow_split_idx < num_flows]

    # get flow arrival times by subtracting the cumulative arrival time of the corresponding potential matrix entry
    arrival_time_sub = np.zeros_like(ia_time_cumsum)
    arrival_time_sub_idx, inv_idx = np.unique(cum_flow_split_idx, return_inverse=True)
    arrival_time_sub_vals = np.bincount(inv_idx, weights=pm_flat_pruned)
    arrival_time_sub[arrival_time_sub_idx] = arrival_time_sub_vals
    arrival_time_sub = np.cumsum(arrival_time_sub)
    flow_arrival_times = ia_time_cumsum - arrival_time_sub
    flow_arrival_times = (total_time * flow_arrival_times) / np.repeat(pm_flat, num_flows_per_pm_entry)

    LOG_CONSOLE_DEBUG(f"Got flow arrival times")

    # the resulting arrival times should be within the total time,
    # and monotonically increasing within the splits per src-dst pair
    if check:
        assert np.all(flow_arrival_times < total_time)
        assert (np.sum(
            np.concatenate([np.diff(m, prepend=0) for m in np.split(flow_arrival_times, cum_flow_split_idx)]) < 0)
                == 0)

    # get flow sizes, types and datarates
    flow_sizes = sample_flow_sizes(num_flows)
    flow_is_tcp = rng.random(num_flows) < prob_tcp
    flow_datarates = get_datarates(rng, flow_sizes, flow_is_tcp)

    # get source and destination per flow
    coordinates = np.indices(potential_matrix.shape).flatten().reshape(2, -1).T  # shape: [n*n, 2]
    flow_coordinates = np.repeat(coordinates, num_flows_per_pm_entry, axis=0)  # shape: [num_flows, 2]

    LOG_CONSOLE_DEBUG(f"Got flow sizes, types, datarates and coordinates")

    # create flow events from the generated data and insert them into the provided event container
    for i in range(len(flow_coordinates)):
        src, dst = flow_coordinates[i]
        arrival_time = flow_arrival_times[i]
        new_flow_arrival = FlowArrival(t=arrival_time, src=src, dst=dst, amount=flow_sizes[i],
                                 datarate=flow_datarates[i], is_tcp=flow_is_tcp[i])
        flow_arrival_timestep = int(arrival_time / ms_per_step)
        events[flow_arrival_timestep].append(new_flow_arrival)
