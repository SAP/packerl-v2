"""
Traffic flow arrivals are generated here, based on the configuration provided in the scenario configuration.
"""
from typing import List, Optional, Dict

import numpy as np

from .flow_arrival import FlowArrival
from .potential_matrix import generate_flows_from_potential_matrix, PotentialMatrixConfig
from .plf import generate_flows_plf, PLFConfig
from .tm_flowparams import generate_flows_tm_flowparams, TMFlowParamsConfig
from src.scenario.config import BaseConfig, SingleOrChoice
from src.scenario.events.event import Event



traffic_config_classes = {
    'tm_flowparams': TMFlowParamsConfig,
    'plf': PLFConfig,
    'potential_matrix': PotentialMatrixConfig,
}


class TrafficConfig(BaseConfig):
    """
    Configuration for traffic flow event generation.
    """
    prob_tcp: SingleOrChoice[float]
    max_flow_size: int  # maximum size of a flow in bytes
    mode: str  # traffic generation mode: 'tm_flowparams', 'plf', 'potential_matrix'
    mode_configs: Dict[str, BaseConfig] = {}

    def __init__(self, **data):

        mode_configs = data.pop('mode_configs', {})
        super().__init__(**data)
        for traffic_mode, cfg_data in mode_configs.items():
            if traffic_mode not in traffic_config_classes.keys():
                raise ValueError(f"Invalid traffic mode: {traffic_mode}")
            self.mode_configs[traffic_mode] = traffic_config_classes[traffic_mode](**cfg_data)


def generate_traffic_flows(G, events: List[List[Event]], scenario_cfg: dict, rng: np.random.Generator, check=False):
    """
    Generate traffic flow events based on the configuration provided in the scenario configuration.
    This is a wrapper function that calls generate_traffic_matrix and generate_traffic_events.
    """
    # if mode is static dot not generate traffic events
    mode = scenario_cfg['events']['traffic']['mode']
    if mode == 'tm_flowparams':
        generate_flows_tm_flowparams(G, events, scenario_cfg, rng)
    elif mode == 'plf':
        plf_config = scenario_cfg['topology']['graph']['mode_configs']['plf']
        plf_depth = plf_config['inner_depth']
        plf_bisection_bandwith = plf_config['bisection_bandwidth']
        generate_flows_plf(events, scenario_cfg, plf_depth, plf_bisection_bandwith)
    elif mode == 'potential_matrix':
        generate_flows_from_potential_matrix(G, events, scenario_cfg, rng, check)
    else:
        raise ValueError(f"Unknown traffic generation mode: {mode}")
