from src.algorithm.algorithm import Algorithm
from src.utils.observations import Observation


class BaselineAlgorithm(Algorithm):
    """
    Abstract class for baseline routing optimization algorithms.
    """
    pass


class BaselineCentralizedAlgorithm(BaselineAlgorithm):
    """
    A baseline algorithm class that provides routing actions in a centralized manner
    (a single actor working on a centralized observation).
    """
    def set_loc(self, actor_loc: str, actor_input: str):
        if actor_loc != 'single':
            raise NotImplementedError("BaselineCentralizedAlgorithm only supports 'single' actor_loc")
        super().set_loc(actor_loc, actor_input)

    def _prepare_obs(self, obs_raw: Observation) -> Observation:
        return obs_raw  # No special preparation needed for centralized baseline
