from src.algorithm.baselines import BaselineCentralizedAlgorithm
from src.packerl_env import PackerlEnv
from src.utils.observations import ObservationGraph
from src.utils.types import ActionRaw, ActionSedi
from src.utils.sp_calculator import ShortestPathCalculator, get_shortest_path_calculator


class SPFCentralizedAlgorithm(BaselineCentralizedAlgorithm):
    """
    Returns a selection of routing edges that corresponds to the shortest paths from sources to destinations,
    calculated by the sp_provider in the respective sp_mode (e.g. EIGRP or OSPF).
    """
    action_raw_representation = "link_weight"

    def __init__(self, env: PackerlEnv, spf_mode: str):
        super().__init__(env)
        self._sp_provider: ShortestPathCalculator = get_shortest_path_calculator(spf_mode)

    def _get_action(self, obs_graph: ObservationGraph) -> (ActionRaw, ActionSedi):
        """
        Overrides the baseline method to return a shortest-path routing.
        Operates on the raw observation, not the filtered observation provided as function argument.
        """
        return self._sp_provider.get_link_weights_and_sedi(obs_graph)
