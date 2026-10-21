"""
This file contains a PackeRL algorithm wrapping the MAGNNETO baseline to make it usable in the PackeRL environment.
"""
from pathlib import Path
import tempfile
import warnings

import networkx as nx
import numpy as np
from torch import Tensor

try:
    import tensorflow as tf
except ImportError as e:
    raise RuntimeError(f"Error importing tensorflow: {e.msg}. Please make sure you've got it installed.")
import torch

from src.algorithm.baselines import BaselineCentralizedAlgorithm
from src.algorithm.baselines.magnneto.actor import Actor as MagnnetoActor
from src.algorithm.baselines.magnneto.environment import Environment as MagnnetoEnv
from src.algorithm.baselines.magnneto.magnneto_utils import traffic_matrix_to_str, obs_to_str
from src.utils.features import edge_datarate_idx, edge_delay_idx, edge_feat_in_struct_order
from src.utils.logging import LOG_CONSOLE_INFO
from src.utils.observations import ObservationGraph
from src.utils.types import ActionRaw, ActionSedi
from src.utils.sp_calculator import ShortestPathCalculator


class MagnnetoShortestPathCalculator(ShortestPathCalculator):
    """
    Custom shortest path calculator for the MAGNNETO algorithm, using the custom weights as the weight.
    """
    sp_mode = "magnneto"
    weight_feat_position = 0  # use the first column as the weight

    def __init__(self):
        super().__init__()
        LOG_CONSOLE_INFO("MagnnetoShortestPathCalculator initialized")

    def _calculate_weights(self, edge_attr: Tensor) -> Tensor:
        return edge_attr[:, self.weight_feat_position]  # use the first column as the weight


class MagnnetoCentralizedAlgorithm(BaselineCentralizedAlgorithm):
    """
    Wrapper class to use the MAGNNETO baseline in the PackeRL environment.
    """
    env_type = "PackeRL"
    traffic_profile = "gravity"
    routing = "sp"
    action_raw_representation = "link_weight"

    def __init__(self, env, magnneto_seed: int, capacity_scaling: float):
        """
        Initializes the MAGNNETO algorithm wrapper
        """
        super().__init__(env)

        magnneto_model_path = (Path(__file__).parent.parent.parent.parent.parent
                               / "data" / "magnneto" / "saved_models" / f"seed{magnneto_seed}" / "actor")
        if not magnneto_model_path.exists():
            raise RuntimeError(f"Could not find magnneto model for seed {magnneto_seed}. "
                               f"Expected it at {magnneto_model_path}. "
                               f"Please make sure that the model exists at the correct location.")
        self._model_path = magnneto_model_path
        self._capacity_scaling = capacity_scaling
        self._magnneto_sp_provider = MagnnetoShortestPathCalculator()

        self._actor = None
        self._cached_topology = None

    def _get_magnneto_weights(self, obs: ObservationGraph) -> dict:
        """
        Consult the MAGNNETO baseline to get the link weights for the given input graph.
        """
        # calling MAGNNETO requires reading topology/traffic files from disk,
        # so use a temporary directory and serialize the topology/traffic there
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            graph_path = temp_path / self.env_type / self.traffic_profile
            graph_path.mkdir(parents=True)

            # write capacities file
            capacities_path = graph_path / "capacities"
            capacities_path.mkdir()
            capacity_fp = str((capacities_path / "graph.txt").resolve())
            with open(capacity_fp, 'w') as capacities_file:
                serialized_topology = obs_to_str(obs, edge_datarate_idx, edge_delay_idx,
                                                 capacity_scaling=self._capacity_scaling, as_int=True)
                capacities_file.write(serialized_topology)

            # write traffic file
            tm_path = graph_path / "TM"
            tm_path.mkdir()
            with open(str((tm_path / "TM-0").resolve()), 'w') as tm_file:
                serialized_traffic = traffic_matrix_to_str(self._env.past_traffic_matrix, obs.agent_ids, as_int=True)
                tm_file.write(serialized_traffic)

            # create MAGNNETO env
            magnneto_env = MagnnetoEnv(env_type=self.env_type,
                                       traffic_profile=self.traffic_profile,
                                       routing=self.routing,
                                       base_dir=temp_dir)

            # (re-)set actor/cached topology only at the start, or when we see a new topology
            if (self._actor is None
                    or self._cached_topology is None
                    or not obs.topology_equal(self._cached_topology)):
                with warnings.catch_warnings():  # ignore TF initializer warnings since we load a model directly after
                    warnings.simplefilter("ignore", category=UserWarning)
                    self._actor = MagnnetoActor(magnneto_env.G, num_features=magnneto_env.num_features)
                    self._actor.build()
                    model = tf.keras.layers.TFSMLayer(self._model_path.resolve(), call_endpoint='serving_default')
                    for w_model, w_actor in zip(model.trainable_variables, self._actor.trainable_variables):
                        w_actor.assign(w_model)
                    self._cached_topology = obs

            # reset MAGNNETO env (this is done every PackeRL step because in MAGNNETO,
            # an episode equals optimizing for a single TM)
            magnneto_env.initialize_environment(num_sample=0, random_env=False)
            magnneto_env.reset(change_sample=False)
            state = magnneto_env.get_state()

            # run one full optimization (imitates MAGNNETO's PPOAgent)
            horizon = int(np.floor(magnneto_env.n_links * 2.75))  # that's their default
            for i in range(horizon):
                logits = self._actor(state)
                actions = [np.argmax(logits.numpy())]
                for action in actions:
                    next_state, reward = magnneto_env.step(action)
                state = next_state

        return nx.get_edge_attributes(magnneto_env.G, "weight")

    def _get_action(self, obs_graph: ObservationGraph) -> (ActionRaw, ActionSedi):
        """
        Get the action from the MAGNNETO baseline, consulting the original MAGNNETO "environment" and its actor
        to get link weights for the shortest path/routing action calculator.
        """

        # get the link weights from MAGNNETO
        magnneto_weights = self._get_magnneto_weights(obs_graph)

        # create an input graph for the shortest path calculator that contains only the MAGNNETO link weights
        sp_input_graph = obs_graph.clone()
        N = sp_input_graph.num_nodes
        flat_edge_index = sp_input_graph.edge_index[0] * N + sp_input_graph.edge_index[1]
        sp_input_attr = torch.empty((sp_input_graph.num_edges, len(edge_feat_in_struct_order)), dtype=torch.float32)
        for (u, v), weight in magnneto_weights.items():
            sp_input_attr[flat_edge_index == (u * N + v), self._magnneto_sp_provider.weight_feat_position] = weight
        sp_input_graph.edge_attr = sp_input_attr

        return self._magnneto_sp_provider.get_link_weights_and_sedi(sp_input_graph)
