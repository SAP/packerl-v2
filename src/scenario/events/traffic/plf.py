from typing import List

import numpy as np

from src.scenario.config import BaseConfig
from src.scenario.events.event import Event
from src.scenario.events.traffic.flow_arrival import FlowArrival
from src.utils.constants import ETHERNET_MTU, IPV4_HEADER_SIZE, UDP_HEADER_SIZE


UDP_PAYLOAD_SIZE = ETHERNET_MTU - (IPV4_HEADER_SIZE + UDP_HEADER_SIZE)


class PLFConfig(BaseConfig):
    pass


def generate_flows_plf(events: List[List[Event]],
                       scenario_cfg: dict,
                       plf_depth,
                       plf_bisection_bandwith):
    """
    Generate flows for Pyramid Lattice Funnel topology. We generate a number of flows between the two hosts
    that corresponds to the number of paths between them. Each flow has a size such that it will last
    throughout the whole scenario, and a bitrate such that all flows together use the full bisection bandwidth.
    The flows are spaced out in time such that combined packet arrival rate corresponds to the bisection bandwidth.
    """
    # num_flows = num_paths_to_destination, which for the Pyramid Lattice Funnel graphs is exponential in layers.
    # Currently, we are limited to 65533 flows per src-dst pair because we don't have unlimited port numbers.
    # Therefore we limit the number of flows for the PLF-16 case, which has 65536 paths, to 65533
    # (the missing 3 out of 65536 flows shouldn't dilute the result too much). Larger depths we simply forbid.
    if plf_depth > 16:
        raise NotImplementedError("PLF depth > 16 not supported due to port number limitations.")
    num_flows = np.minimum(2 ** plf_depth, 65533)

    ms_per_step = scenario_cfg['ms_per_step']

    # calculate flow size such that we send data throughout the whole scenario
    scenario_duration_ms = ms_per_step * len(events)
    flow_bytes = int(scenario_duration_ms * plf_bisection_bandwith / (8 * 1000))

    # calculate flow bitrate such that all flows together use the bisection bandwidth
    flow_bitrate = (plf_bisection_bandwith / num_flows).astype(int).item()

    # the bisection bandwidth (in bps) tells us how much total bandwidth all flows will use.
    # -> calculate total number of packets (bits // 8 // MTU) per second
    # -> calculate packet inter-arrival time, which will be used to space the arrival time of each flow
    global_packet_rate = plf_bisection_bandwith / (8 * UDP_PAYLOAD_SIZE)  # packets per second
    global_packet_interarrival_time = 1000 / global_packet_rate  # ms
    flow_arrival_times = np.arange(num_flows, dtype=np.float32) * global_packet_interarrival_time
    # cut off flows that would arrive after the scenario ends
    flow_arrival_times = flow_arrival_times[flow_arrival_times < scenario_duration_ms].tolist()

    for arrival_time in flow_arrival_times:
        new_flow = FlowArrival(
            t=arrival_time,
            src=0,
            dst=1,
            amount=flow_bytes,
            datarate=flow_bitrate,
            is_tcp=False
        )
        events[int(arrival_time // ms_per_step)].append(new_flow)
