import torch
from torch import Tensor
import torch.nn.functional as F

from src.utils.logging import LOG_CONSOLE_FUNCTION
from src.utils.types import LogprobTensor, ValueTensor, EntropyTensor, AdvantageTensor, ReturnTensor, RatioTensor


def policy_loss(normalized_adv: AdvantageTensor, ratio: RatioTensor, clip_range: float) -> Tensor:
    """
    Clipped surrogate total_loss for the policy. Note that we return the negative loss.
    """
    LOG_CONSOLE_FUNCTION()

    policy_loss_unclipped = normalized_adv * ratio
    policy_loss_clipped = normalized_adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped).mean()
    return policy_loss


def value_loss(returns: ReturnTensor, train_values: ValueTensor, rollout_values: ValueTensor, clip_range: float) -> Tensor:
    """
    Value loss using the TD(gae_lambda) target for the value function.
    """
    LOG_CONSOLE_FUNCTION()

    value_loss = F.mse_loss(returns, train_values)
    if clip_range > 0:
        # In OpenAI's PPO implementation, the value function is clipped around the previous value estimate
        # the worst of the clipped and unclipped versions is then used to train the value function
        value_clipped = rollout_values + (train_values - rollout_values).clamp(-clip_range, clip_range)
        value_loss_clipped = F.mse_loss(returns, value_clipped)
        value_loss = torch.max(value_loss, value_loss_clipped)
    return value_loss


def entropy_loss(entropy: EntropyTensor, logprob: LogprobTensor) -> Tensor:
    """
    Entropy loss to favor exploration
    """
    LOG_CONSOLE_FUNCTION()

    if entropy is None:  # Approximate entropy when no analytical form is available
        entropy_loss = -torch.mean(-logprob)
    else:
        entropy_loss = -torch.mean(entropy)
    return entropy_loss
