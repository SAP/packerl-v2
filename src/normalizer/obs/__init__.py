from src.normalizer.obs._obs_normalizer import ObservationNormalizer
from src.normalizer.obs.dummy import DummyObservationNormalizer
from src.normalizer.obs.rms import RMSObservationNormalizer


def get_obs_normalizer(normalize_observations: bool, feature_counts: dict) -> ObservationNormalizer:
    """
    Get the observation normalizer according to the configuration.
    """
    if normalize_observations:
        return RMSObservationNormalizer(num_node_features=feature_counts["node"],
                                        num_edge_features=feature_counts["edge"],
                                        num_global_features=feature_counts["global"])
    else:
        return DummyObservationNormalizer(num_node_features=feature_counts["node"],
                                        num_edge_features=feature_counts["edge"],
                                        num_global_features=feature_counts["global"])
