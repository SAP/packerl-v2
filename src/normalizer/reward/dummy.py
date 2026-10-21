from src.normalizer.reward._reward_normalizer import RewardNormalizer
from src.utils.types import Reward


class DummyRewardNormalizer(RewardNormalizer):
    """
    Dummy reward normalizer that does nothing.
    """
    def __init__(self, reward_clip = 5, epsilon = 1e-10):
        super().__init__(reward_clip, epsilon)

    def reset(self, _):
        pass

    def update_and_normalize(self, reward: Reward):
        return reward

    def _update(self, reward: Reward):
        pass

    def normalize(self, reward: Reward):
        return reward
