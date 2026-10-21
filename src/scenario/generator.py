from copy import deepcopy
from typing import List

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .scenario import NetworkScenario, ScenarioConfig
from .topology import generate_topology
from .events import generate_events


class Generator:
    """
    The Generator class is used to generate network scenarios based on a given configuration.
    It holds the current random number generator (rng) and the current scenario configuration set.
    From this configuration set, it can generate a network scenario by sampling concrete values for the configuration.
    """
    def __init__(self, scenario_config: DictConfig):
        """
        Initializes the generator with a given set of presets and custom configuration.
        """
        scen_cfg = OmegaConf.to_container(scenario_config, resolve=True)
        self.num_topologies = scen_cfg.pop("num_topologies")
        self.num_event_seqs_per_topology = scen_cfg.pop("num_event_seqs_per_topology")
        self.num_event_seq_repeats = scen_cfg.pop("num_event_seq_repeats")
        self.vis_n_topologies = scen_cfg.pop("vis_n_topologies")
        self.vis_n_event_seqs = scen_cfg.pop("vis_n_event_seqs")

        self.buffer_size_bdp = scen_cfg.pop("buffer_size_bdp")  # currently handled in ns-3
        self.prob_tcp = float(scen_cfg["events"]["traffic"]["prob_tcp"])  # currently handled in ns-3
        self.use_tcp_sack = scen_cfg.pop("use_tcp_sack")  # currently handled in ns-3
        scen_cfg["scenario_length"] = scen_cfg.pop("ep_length")
        self.ep_length = scen_cfg["scenario_length"]
        self.ms_per_step = scen_cfg["ms_per_step"]  # currently handled in ns-3

        self.scenario_cfg_set = ScenarioConfig(**scen_cfg)

        self.rng: np.random.Generator | None = None  # initialized in reset_rng()
        self.rng_topology_id: int = 0  # counts how many networks have been generated so far
        self.reset_rng()

    def reset_rng(self):
        self.rng = np.random.default_rng(self.scenario_cfg_set.seed)
        self.rng_topology_id = 0

    def generate(self, with_events: bool = True, topology_multiplier: int = 1) -> List[NetworkScenario]:
        """
        Generates multiple network scenarios as specified by num_episodes.

        Args:
            with_events (bool): Whether to generate events (traffic, link failures, etc.) for the scenarios.
            topology_multiplier (int): Multiplier for the number of topologies to generate.
        """
        scenarios = []
        num_topologies = self.num_topologies * topology_multiplier
        for i in range(num_topologies):

            # sample scenario config
            scenario_cfg = self.scenario_cfg_set.sample(self.rng)
            # create network topology (= topology graph and attributes)
            G_ = generate_topology(scenario_cfg, self.rng, self.rng_topology_id)
            topology_onehot = np.zeros(num_topologies, dtype=np.float32)
            topology_onehot[i] = 1.0

            for j in range(self.num_event_seqs_per_topology):

                # create events (traffic and e.g. link failures)
                if with_events:
                    events_ = generate_events(G_, scenario_cfg, self.rng)
                else:
                    events_ = [[] for _ in range(self.ep_length)]
                event_seq_onehot = np.zeros(self.num_event_seqs_per_topology, dtype=np.float32)
                event_seq_onehot[j] = 1.0

                for k in range(self.num_event_seq_repeats):

                    # copy topology/events to avoid shared references
                    G = G_.copy()
                    events = deepcopy(events_)
                    visualize_scenario = i < self.vis_n_topologies and j < self.vis_n_event_seqs and k == 0
                    scenario = NetworkScenario(G, events,
                                               self.ms_per_step,
                                               self.buffer_size_bdp,
                                               self.prob_tcp,
                                               self.use_tcp_sack,
                                               onehot_id=np.concatenate([topology_onehot, event_seq_onehot]),
                                               visualize=visualize_scenario)
                    scenarios.append(scenario)

            self.rng_topology_id += 1

        return scenarios
