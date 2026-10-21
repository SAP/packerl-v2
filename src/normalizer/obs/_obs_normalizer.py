from src.normalizer._normalizer import AbstractNormalizer


class ObservationNormalizer(AbstractNormalizer):
    """
    Abstract class for observation normalizers.
    """
    def __init__(self, num_node_features: int, num_edge_features: int, num_global_features: int,
                 obs_clip = 10, epsilon = 1e-10):
        super().__init__(epsilon)
        self.obs_clip = obs_clip

    def to_canonical_hash_str(self) -> str:
        raise NotImplementedError
