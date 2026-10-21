import torch

from src.normalizer.reward._reward_normalizer import RewardNormalizer
from src.normalizer.running_mean_std import TorchRunningMeanStd
from src.utils.constants import BIRDSEYE_OBSERVER_ID
from src.utils.types import Reward


class RMSRewardNormalizer(RewardNormalizer):
    """
    Reward normalizer that uses the RunningMeanStd approach. Note that the reward is supplied per-agent as a tensor
    within the observation, with the value for observer_ids==-1 being the global reward.

    Uses one normalizer instance for the global rewards, and one shared instance for the rewards of each agent.
    The shared instance is updated with the mean of the local rewards of all agents.
    """
    def __init__(self, discount_factor, reward_clip = 10, epsilon = 1e-10):
        super().__init__(reward_clip, epsilon)

        self.discount_factor = discount_factor

        self.global_normalizer = TorchRunningMeanStd(epsilon=epsilon, shape=(1,))
        self.local_normalizer = TorchRunningMeanStd(epsilon=epsilon, shape=(1,))

        self.global_running_ret = torch.tensor([0.0], dtype=self.global_normalizer.dtype)
        self.local_running_mean_ret = torch.tensor([0.0], dtype=self.local_normalizer.dtype)

    def reset(self, _):
        """
        Reset the global and local running returns
        """
        self.global_running_ret = torch.tensor([0.0], dtype=self.global_normalizer.dtype)
        self.local_running_mean_ret = torch.tensor([0.0], dtype=self.local_normalizer.dtype)

    def update_and_normalize(self, reward: Reward) -> Reward:
        """
        Update the running returns and normalize the rewards for all agents (incl. global)
        """
        self._update(reward)
        reward = self.normalize(reward)
        return reward

    def _update(self, reward: Reward):
        """
        Update the running returns for all agents (incl. global)
        """
        reward_values, reward_observer_idx = reward
        is_global = reward_observer_idx == BIRDSEYE_OBSERVER_ID
        global_reward = reward_values[is_global]
        self.global_running_ret = self.discount_factor * self.global_running_ret + global_reward
        self.global_normalizer.update(self.global_running_ret.to(self.global_normalizer.dtype))

        local_rewards = reward_values[~is_global]
        self.local_running_mean_ret = self.discount_factor * self.local_running_mean_ret + torch.mean(local_rewards)
        self.local_normalizer.update(self.local_running_mean_ret.to(self.local_normalizer.dtype))

    def normalize(self, reward: Reward) -> Reward:
        """
        Normalize the rewards for all agents (incl. global)
        """
        reward_values, reward_observer_idx = reward
        is_global = reward_observer_idx == BIRDSEYE_OBSERVER_ID
        global_reward = reward_values[is_global].to(self.global_normalizer.dtype)
        scaled_global_reward = (global_reward - self.global_normalizer.mean) / torch.sqrt(self.global_normalizer.var + self.epsilon)
        scaled_global_reward = torch.clip(scaled_global_reward, -self.reward_clip, self.reward_clip)

        local_rewards = reward_values[~is_global].to(self.local_normalizer.dtype)
        scaled_local_rewards = (local_rewards - self.local_normalizer.mean) / torch.sqrt(self.local_normalizer.var + self.epsilon)
        scaled_local_rewards = torch.clip(scaled_local_rewards, -self.reward_clip, self.reward_clip)

        # Update the reward values with the scaled rewards
        reward_values_normalized = torch.empty_like(reward_values, dtype=torch.float)
        reward_values_normalized[is_global] = scaled_global_reward.float()
        reward_values_normalized[~is_global] = scaled_local_rewards.float()
        return reward_values_normalized, reward_observer_idx
