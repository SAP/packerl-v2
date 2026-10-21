from typing import List, Optional

import networkx as nx
import numpy as np

from src.scenario.config import BaseConfig
from src.scenario.events.event import Event
from src.scenario.events.traffic.flow_arrival import FlowArrival
from collections import defaultdict
import json


MIN_FLOW_SIZE = 10
MAX_DATARATE = 1e9  # 1 Gbps


class TMFlowParamsConfig(BaseConfig):
    """
    Configuration for traffic matrix based flow generation using provided TM and flow parameters.
    """
    tm_path: Optional[str]  # path to the traffic matrix file, if provided
    flowparams_path: Optional[str]  # path to the flow parameters
    interarrival_time_scaling: int  # interarrival time scaling
    flowsize_scaling: float  # flow size scaling
    flow_datarate_scaling: float  # flow data rate scaling


def generate_flows_s_t(events: List[List[Event]],
                       rng: np.random.Generator,
                       flow_params: dict,
                       prob_mice: float,
                       prob_tcp: float,
                       target_demand: int,
                       max_rel_arrival_time: float,
                       ms_per_step: float,
                       max_flow_size: int,
                       scenario_iat_scale: float,
                       scenario_flowsize_scale: float,
                       scenario_datarate_scale: float,
                       srcindex: int,
                       dstindex: int):

    mice_size_sig, _, mice_size_scale = flow_params['mice']['data']
    mice_iat_sig, _, mice_iat_scale = flow_params['mice']['iat']
    eleph_size_a, eleph_size_loc, eleph_size_scale = flow_params['elephant']['data']
    eleph_iat_sig, _, eleph_iat_scale = flow_params['elephant']['iat']
    udp_datarate_a, udp_datarate_loc, udp_datarate_scale = flow_params['udp']['datarates']

    def sample_flow():
        if rng.random() < prob_mice:
            # Sample mice flow
            size = rng.lognormal(sigma=mice_size_sig,
                                 mean=np.log(mice_size_scale * scenario_flowsize_scale), size=1)[0]
            interarrival_time = rng.lognormal(sigma=mice_iat_sig,
                                              mean=np.log(mice_iat_scale * scenario_iat_scale), size=1)[0]
        else:
            # Sample elephant flow
            size = ((rng.pareto(eleph_size_a, size=1)[0] + 1) * eleph_size_scale * scenario_flowsize_scale
                    + eleph_size_loc)
            interarrival_time = rng.lognormal(sigma=eleph_iat_sig,
                                              mean=np.log(eleph_iat_scale * scenario_iat_scale), size=1)[0]

        is_tcp = rng.random() < prob_tcp
        if is_tcp:
            datarate = 0
        else:
            datarate = ((rng.pareto(udp_datarate_a, size=1)[0] + 1) * udp_datarate_scale * scenario_datarate_scale
                        + udp_datarate_loc)

        # clamp
        size_clamped = max(min(size, max_flow_size), MIN_FLOW_SIZE)
        datarate_clamped = 0 if is_tcp else max(min(datarate, MAX_DATARATE), 1)

        return interarrival_time, size_clamped, is_tcp, datarate_clamped

    iats, sizes, is_tcps, drs = [], [], [], []
    i, cum_datarate = 0, 0

    # Iterate until datarate matches target demand or max_time is reached (i.e. we exceed the number of scenario steps)
    while True:
        iat, size, is_tcp, dr = sample_flow()

        iats.append(iat)
        sizes.append(size)
        is_tcps.append(is_tcp)
        drs.append(dr)

        cum_datarate += dr
        i += 1
        if cum_datarate >= target_demand:
            break

    arrival_times = np.array(iats).cumsum()
    sizes = np.round(np.array(sizes)).astype(int)
    is_tcps = np.array(is_tcps, dtype=bool)
    drs = np.round(np.array(drs)).astype(int)

    # re-scale arrival times to fit within max_rel_arrival_time * max_time
    cur_last_arrival = arrival_times[-1]
    desired_last_arrival = max_rel_arrival_time * len(events) * ms_per_step
    arrival_times *= desired_last_arrival / cur_last_arrival

    for arrival_time, size, is_tcp, dr in zip(arrival_times, sizes, is_tcps, drs):
        new_flow = FlowArrival(
            t=arrival_time,
            src=srcindex,
            dst=dstindex,
            amount=size,
            datarate=dr,
            is_tcp=is_tcp
        )
        events[int(arrival_time // ms_per_step)].append(new_flow)

    return events


def generate_flows_tm_flowparams(G: nx.Graph, events: List[List[Event]], scenario_cfg: dict, rng: np.random.Generator):
    traffic_cfg = scenario_cfg['events']['traffic']
    tmf_config = traffic_cfg['mode_configs']['tm_flowparams']

    # Load and validate traffic matrix
    tm_path = tmf_config.get('tm_path', None)
    if tm_path is None:
        raise ValueError("Traffic matrix path must be specified in the configuration.")
    tm = np.loadtxt(tm_path, delimiter=' ')
    if tm.shape != (G.number_of_nodes(), G.number_of_nodes()):
        raise ValueError(f"Traffic matrix shape {tm.shape} does not match graph's num_nodes {G.number_of_nodes()}.")

    # Load and validate flow distribution parameters
    flowparams_path = tmf_config.get('flowparams_path', None)
    if flowparams_path is None:
        raise ValueError("Flow distribution parameters must be specified in the configuration.")
    flowparams = defaultdict()
    with open(flowparams_path, 'r') as f:
        flowparams = json.load(f)

    # traverse all src-dst pairs in the tm and generate flows
    for srcindex in range(tm.shape[0]):
        for dstindex in range(tm.shape[1]):
            if srcindex == dstindex:
                continue  # skip self-loops
            demand_value = tm[srcindex, dstindex]
            if demand_value <= 0:
                continue  # skip zero demand

            generate_flows_s_t(
                events,
                rng,
                flow_params=flowparams,
                prob_mice=0.9,
                prob_tcp=traffic_cfg['prob_tcp'],
                target_demand=int(demand_value),
                max_rel_arrival_time=0.5,
                ms_per_step=scenario_cfg['ms_per_step'],
                max_flow_size=traffic_cfg['max_flow_size'],
                scenario_iat_scale=tmf_config['interarrival_time_scaling'],
                scenario_flowsize_scale=tmf_config['flowsize_scaling'],
                scenario_datarate_scale=tmf_config['flow_datarate_scaling'],
                srcindex=srcindex,
                dstindex=dstindex
            )
