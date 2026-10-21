"""
Topology graph generation is done here, based on the configuration provided in the scenario configuration.
"""
from typing import Dict, Any

import networkx as nx
import numpy as np

from .barabasi_albert import generate_barabasi_albert, BarabasiAlbertConfig
from .erdos_renyi import generate_erdos_renyi, ErdosRenyiConfig
from .magnneto import generate_magnneto, MagnnetoDatasetConfig
from .plf import generate_plf, PLFConfig
from .static_dot import DOT_TRAFFIC_SCALING, generate_static_dot, StaticDotConfig
from .topology_zoo import generate_topology_zoo, TopologyZooConfig, TZ_TRAFFIC_SCALING
from .watts_strogatz import generate_watts_strogatz, WattsStrogatzConfig
from src.scenario.config import BaseConfig, SingleOrChoice
from src.utils.logging import LOG_CONSOLE_DEBUG, LOG_CONSOLE_FUNCTION

graph_generators = {
    "barabasi_albert": generate_barabasi_albert,
    "erdos_renyi": generate_erdos_renyi,
    "magnneto": generate_magnneto,
    "plf": generate_plf,
    "static_dot": generate_static_dot,
    "topology_zoo": generate_topology_zoo,
    "watts_strogatz": generate_watts_strogatz,
}

graph_config_classes = {
    "barabasi_albert": BarabasiAlbertConfig,
    "erdos_renyi": ErdosRenyiConfig,
    "magnneto": MagnnetoDatasetConfig,
    "plf": PLFConfig,
    "static_dot": StaticDotConfig,
    "topology_zoo": TopologyZooConfig,
    "watts_strogatz": WattsStrogatzConfig,
}

graph_traffic_scalings = {
    "barabasi_albert": lambda G: np.maximum(1, 22 * np.sqrt(G.number_of_nodes() + 10) - 75),
    "erdos_renyi": lambda G: np.maximum(1, 16 * np.sqrt(G.number_of_nodes()) - 30),
    "plf": lambda G: 1,
    "static_dot": lambda G: DOT_TRAFFIC_SCALING.get(G.graph['name'], 1),
    "topology_zoo": lambda G: TZ_TRAFFIC_SCALING[G.graph['name']],
    "watts_strogatz": lambda G: np.maximum(1, 22 * np.sqrt(G.number_of_nodes() + 10) - 75),
    "default": lambda G: 10,
}


class TopologyGraphConfig(BaseConfig):
    """
    Configuration for topology graph generation.
    """
    min_nodes: int
    max_nodes: int
    mode: SingleOrChoice[str]
    mode_configs: Dict[str, BaseConfig] = {}

    def __init__(self, **data):

        mode_configs = data.pop('mode_configs', {})
        super().__init__(**data)
        for graph_mode, cfg_data in mode_configs.items():
            if graph_mode not in graph_generators.keys():
                raise ValueError(f"Invalid graph mode: {graph_mode}")
            if graph_mode in graph_config_classes.keys():
                self.mode_configs[graph_mode] = graph_config_classes[graph_mode](**cfg_data)

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        sampled_cfg = super().sample(rng)
        min_nodes = sampled_cfg.pop('min_nodes')
        max_nodes = sampled_cfg.pop('max_nodes')
        sampled_cfg['num_nodes'] = rng.integers(min_nodes, max_nodes, endpoint=True)
        return sampled_cfg


def generate_topology_graph(scenario_cfg: dict, rng: np.random.Generator, rng_i: int) -> nx.Graph:
    """
    Generate a random graph based on the given scenario configuration.
    """
    LOG_CONSOLE_FUNCTION()

    # preparation
    graph_cfg = scenario_cfg['topology']['graph']
    seed = scenario_cfg['seed']
    num_nodes = graph_cfg['num_nodes']
    chosen_mode = graph_cfg['mode']
    if chosen_mode not in graph_generators.keys():
        raise ValueError(f"Invalid chosen topology graph mode: {chosen_mode}")
    graph_generator_cfg = graph_cfg['mode_configs'].get(chosen_mode, {})
    graph_generator = graph_generators[chosen_mode]

    # generate graph depending on the chosen graph mode
    G = graph_generator(num_nodes=num_nodes, rng=rng, rng_i=rng_i, **graph_generator_cfg)

    LOG_CONSOLE_DEBUG(f"Generated graph {G}")

    # assign graph-dependent attributes
    if G.graph.get('traffic_scaling', None) is None:
        scaling_method = graph_traffic_scalings.get(chosen_mode, graph_traffic_scalings['default'])
        G.graph['traffic_scaling'] = scaling_method(G)
    graph_name = f"{chosen_mode}-seed{seed}-i{rng_i}"
    if G.graph.get('name') is not None:
        graph_name += f"-{G.graph['name']}"
    G.graph['name'] = graph_name

    return G
