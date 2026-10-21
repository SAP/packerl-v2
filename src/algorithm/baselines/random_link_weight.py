import networkx as nx
import torch

from src.algorithm.baselines import BaselineCentralizedAlgorithm
from src.utils.observations import ObservationGraph
from src.utils.sp_calculator import apsp_to_action_sedi
from src.utils.types import ActionRaw, ActionSedi


class RandomLinkWeightCentralizedAlgorithm(BaselineCentralizedAlgorithm):
    """
    Returns routing paths generated with random link weights
    """
    action_raw_representation = "link_weight"

    def _get_action(self, obs_graph: ObservationGraph) -> (ActionRaw, ActionSedi):
        """
        Overrides the baseline method to return a shortest-path routing from randomly generated link weights.
        """
        random_weights = torch.rand(obs_graph.num_edges) * 4 + 1

        # all-pairs-shortest-paths computation requires a networkx graph
        G = nx.DiGraph()
        G.add_nodes_from(obs_graph.agent_ids.numpy())
        edges_with_weights = [(u, v, {'randomWeight': int(w.item())})
                              for (u, v), w in zip(obs_graph.edge_index.T.numpy(), random_weights)]
        G.add_edges_from(edges_with_weights)

        # compute all-pairs-shortest-paths with random weights and, from there, obtain the action sedi
        apsp = nx.all_pairs_dijkstra(G, cutoff=None, weight='randomWeight')
        action_sedi = apsp_to_action_sedi(apsp, obs_graph.agent_ids, obs_graph.edge_index)

        # 'raw' actions are the random link weights
        return random_weights, action_sedi
