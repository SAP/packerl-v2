import torch
import torch_scatter

from src.algorithm.baselines import BaselineCentralizedAlgorithm
from src.utils.observations import ObservationGraph
from src.utils.observations.obs_utils import get_nexthop_candidate_idx
from src.utils.types import ActionRaw, ActionSedi


class RandomNextHopCentralizedAlgorithm(BaselineCentralizedAlgorithm):
    """
    Randomly selects a next hop from the available routing edges.
    """
    action_raw_representation = "next_hop"

    def _get_action(self, obs_graph: ObservationGraph) -> (ActionRaw, ActionSedi):
        """
        Overrides the baseline method to return a routing consisting of random next-hop selections
        per routing and destination node.
        """
        random_values = torch.rand((obs_graph.num_nodes * obs_graph.num_edges,))
        _, action_sedi = torch_scatter.scatter_max(random_values, index=get_nexthop_candidate_idx(obs_graph))

        return random_values, action_sedi
