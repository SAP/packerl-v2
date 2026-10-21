from src.normalizer.obs._obs_normalizer import ObservationNormalizer
from src.utils.observations import Observation


class DummyObservationNormalizer(ObservationNormalizer):
    """
    Dummy observation normalizer that does nothing.
    """
    def reset(self, obs: Observation) -> Observation:
        return obs

    def update_and_normalize(self, obs: Observation) -> Observation:
        return obs

    def _update(self, obs: Observation):
        pass

    def normalize(self, obs: Observation) -> Observation:
        return obs

    def to_canonical_hash_str(self) -> str:
        return "dummy (no state)"
