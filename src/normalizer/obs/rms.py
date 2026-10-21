import torch
from torch import Tensor

from src.normalizer.running_mean_std import TorchRunningMeanStd
from src.normalizer.obs._obs_normalizer import ObservationNormalizer
from src.utils.observations import Observation
from src.utils.canonical_bytes import canonical_hash


class RMSObservationNormalizer(ObservationNormalizer):
    """
    This observation normalizer maintains separate running statistics for node, edge and global features.
    """
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 num_global_features: int,
                 observation_clip = 10,
                 epsilon = 1e-10):
        super().__init__(num_node_features, num_edge_features, num_global_features,
                         obs_clip=observation_clip, epsilon=epsilon)

        self.node_normalizer = TorchRunningMeanStd(epsilon=epsilon, shape=(num_node_features,))
        self.edge_normalizer = TorchRunningMeanStd(epsilon=epsilon, shape=(num_edge_features,))
        self.global_normalizer = TorchRunningMeanStd(epsilon=epsilon, shape=(num_global_features,))

    def reset(self, obs: Observation) -> Observation:
        """
        To be called after the environment reset. Updates the normalizer statistics with the
        initial observations and returns the normalized observations.
        """
        self._update(obs)
        return self.normalize(obs)

    def update_and_normalize(self, obs: Observation) -> Observation:
        """
        Update the normalizer statistics with the given observations and return the normalized observations.
        """
        self._update(obs)
        return self.normalize(obs)

    def _update(self, obs: Observation):
        """
        Update the normalizer statistics with the given observations.
        """
        self.node_normalizer.update(obs.x.to(self.node_normalizer.dtype))
        self.edge_normalizer.update(obs.edge_attr.to(self.edge_normalizer.dtype))
        self.global_normalizer.update(obs.u.to(self.global_normalizer.dtype))

    def _normalize(self, val: Tensor, normalizer: TorchRunningMeanStd) -> Tensor:
        """
        Helper to normalize a given observation component using its running statistics.

        Args:
            val: the tensor to normalize.
            normalizer: the corresponding TorchRunningMeanStd instance.

        Returns:
            The normalized tensor.
        """
        scaled_obs_component = (val.to(normalizer.dtype) - normalizer.mean) / torch.sqrt(normalizer.var + self.epsilon)
        scaled_obs_component = torch.clip(scaled_obs_component, -self.obs_clip, self.obs_clip)
        return scaled_obs_component.float()

    def normalize(self, obs: Observation) -> Observation:
        """
        Normalize observations using the current statistics.
        Calling this method does not update statistics. It can thus be called for training as well as evaluation.
        """
        obs.__setattr__("x", self._normalize(obs.x, normalizer=self.node_normalizer))
        obs.__setattr__("edge_attr", self._normalize(obs.edge_attr, normalizer=self.edge_normalizer))
        obs.__setattr__("u", self._normalize(obs.u, normalizer=self.global_normalizer))

        return obs

    def to_canonical_hash_str(self) -> str:
        return canonical_hash([
            self.node_normalizer.to_canonical_hash(),
            self.edge_normalizer.to_canonical_hash(),
            self.global_normalizer.to_canonical_hash(),
            self.obs_clip,
            self.epsilon
        ])
