from src.normalizer._normalizer import AbstractNormalizer


class RewardNormalizer(AbstractNormalizer):
    """
    Abstract class for reward normalizers. Note that the reward is supplied per-agent as a tensor
    within the observation, with the value for observer_ids==-1 being the global reward.
    """
    def __init__(self, reward_clip = 5, epsilon = 1e-10):
        super().__init__(epsilon)
        self.reward_clip = reward_clip
