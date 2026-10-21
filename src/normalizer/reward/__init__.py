from src.normalizer.reward._reward_normalizer import RewardNormalizer
from src.normalizer.reward.dummy import DummyRewardNormalizer
from src.normalizer.reward.rms import RMSRewardNormalizer


def get_reward_normalizer(normalize_rewards, discount_factor) -> RewardNormalizer:
    """
    Get the reward normalizer according to the configuration.
    """
    if normalize_rewards:
        return RMSRewardNormalizer(discount_factor=discount_factor)
    else:
        return DummyRewardNormalizer()
