from typing import NamedTuple, Generator, List, Tuple

import numpy as np
import torch
from torch import Tensor

from src.trainer.ppo.utils.ret_adv import compute_returns_and_advantages
from src.utils.logging import LOG_CONSOLE_FUNCTION
from src.utils.observations import Observation, ObservationBatch
from src.utils.types import ActionRaw, ActionSedi, Reward, LogprobTensor, ValueTensor, AdvantageTensor, ReturnTensor


def explained_variance_function(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    LOG_CONSOLE_FUNCTION()

    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


class Minibatch(NamedTuple):
    """
    Named tuple containing samples from the rollout buffer.
    """
    observations: ObservationBatch
    obs_agent_counts: Tensor
    actions_raw: ActionRaw
    actions_sedi: ActionSedi
    old_logprobs: LogprobTensor
    old_values: ValueTensor
    advantages: AdvantageTensor
    returns: ReturnTensor


class OnPolicyBuffer:
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected using the current policy.
    For multi-agent RL, these transitions may either be collected for each node/agent per timestep, or for each
    full graph/set of agents. This class gives a generic interface for both and is inherited by NodeOnPolicyBuffer
    and GraphOnPolicyBuffer for this distinction.

    The buffer only persists for a single algorithm iteration, i.e., it is used to store recent (on-policy) transitions.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state (again node or graph-wise)
    and the log probability of the taken actions (always node-wise).

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not actions selection.
    """
    def __init__(self,
                 buffer_size: int,
                 num_episodes: int,
                 actor_loc: str,
                 value_loc: str,
                 device: torch.device | str):
        """
        Initializes the rollout buffer.
        """
        self.buffer_size = buffer_size
        self.num_episodes = num_episodes
        self.actor_loc = actor_loc
        self.value_loc = value_loc
        self.device = device

        # initialize all the values to be stored in the buffer
        self.observations = None
        self.actions_raw = None
        self.actions_sedi = None
        self.logprobs = None
        self.values = None
        self.value_observers = None
        self.rewards = None
        self.reward_observers = None
        self.dones = None
        self.terminal_observations = None

        self.bootstrapped_rewards = None
        self.returns = None
        self.advantages = None

        self.current_position = None

        # reset all values to their default value to get ready to collect data
        self.reset()

    @property
    def filled(self) -> bool:
        """
        Returns: True if the buffer is fully filled with data from the outside, False otherwise.
        """
        return (len(self.observations) == self.buffer_size
                and len(self.actions_raw) == self.buffer_size
                and len(self.actions_sedi) == self.buffer_size
                and len(self.logprobs) == self.buffer_size
                and len(self.values) == self.buffer_size
                and len(self.value_observers) == self.buffer_size
                and len(self.rewards) == self.buffer_size
                and len(self.reward_observers) == self.buffer_size
                and len(self.terminal_observations) == self.num_episodes
                )  # advantages and returns will be computed from the buffer, and are not checked here

    @property
    def ready_for_training(self) -> bool:
        """
        Returns: True if the buffer is ready to be used, i.e., it is filled with data from the outside and
         the advantages and returns have been computed.
        """
        return (self.filled
                and len(self.bootstrapped_rewards) == self.buffer_size
                and len(self.returns) == self.buffer_size
                and len(self.advantages) == self.buffer_size)

    def reset(self) -> None:
        LOG_CONSOLE_FUNCTION()

        # data to be placed into the buffer in every rollout step
        self.observations: List[Observation] = []
        self.actions_raw: List[ActionRaw] = []
        self.actions_sedi: List[ActionSedi] = []
        self.logprobs: List[LogprobTensor] = []
        self.values: List[ValueTensor] = []
        self.value_observers: List[Tensor] = []
        self.rewards: List[Tensor] = []
        self.reward_observers: List[Tensor] = []
        self.dones: Tensor = torch.empty(self.buffer_size, device=self.device)  # pre-allocate dones for faster access
        self.terminal_observations: List[Observation] = []

        # data to be computed from the buffer
        self.bootstrapped_rewards: List[Tensor] = []
        self.returns: List[ReturnTensor] = []
        self.advantages: List[AdvantageTensor] = []

        self.current_position = 0

    def add(self,
            obs: Observation,
            action_raw: ActionRaw,
            action_sedi: ActionSedi,
            logprob: LogprobTensor,
            value: (ValueTensor, Tensor),
            reward: Reward,
            done: Tensor,
            terminal_obs: Observation = None):
        LOG_CONSOLE_FUNCTION()

        self.observations.append(obs.to(self.device))
        self.actions_raw.append(action_raw.to(self.device))
        self.actions_sedi.append(action_sedi.to(self.device))
        self.logprobs.append(logprob.to(self.device))
        self.values.append(value[0].to(self.device))
        self.value_observers.append(value[1].to(self.device))  # value is a tuple (values, value_observer_ids)
        self.rewards.append(reward[0].to(self.device))
        self.reward_observers.append(reward[1].to(self.device))  # reward is a tuple (rewards, reward_observer_ids)
        self.dones[self.current_position] = done.to(self.device)  # done is always a scalar tensor
        if terminal_obs is not None:
            self.terminal_observations.append(terminal_obs.to(self.device))

        # keeping track of the number of entries
        self.current_position += 1

    def bootstrap_rewards(self, terminal_values: List[Tensor], discount_factor: float) -> List[Tensor]:
        """
        Bootstraps the rewards with the provided terminal values.
        This is useful when computing returns for incomplete episodes.

        Args:
            terminal_values (List[Tensor]): List of terminal state values for each episode in the buffer.
            discount_factor (float): Discount factor (gamma) for return calculation.
        """
        LOG_CONSOLE_FUNCTION()

        assert len(terminal_values) == self.num_episodes, \
            "Number of terminal value tensors does not match number of episodes in the buffer."

        # bootstrap terminal rewards with provided terminal values: R_t = r_t + gamma * V(s_{t+1}) if done else r_t
        assert len(terminal_values) == self.num_episodes, \
            "Number of terminal value tensors does not match number of episodes in the buffer."
        done_indices = self.dones.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        self.bootstrapped_rewards = self.rewards.copy()
        for i, idx in enumerate(done_indices):
            self.bootstrapped_rewards[idx] += discount_factor * terminal_values[i].to(self.device)

    def compute_returns_and_advantages(self, gae_lambda: float, discount_factor: float):
        """
        Computes the returns and advantages for the entire buffer.
        See called function `compute_returns_and_advantages()` for details.

        Args:
            terminal_values (List[Tensor]): List of terminal state values for each episode in the buffer.
            gae_lambda (float): Lambda parameter for GAE.
            discount_factor (float): Discount factor (gamma) for return calculation.
        """
        assert self.filled, "Buffer is not filled with data yet. Cannot compute returns and advantages."
        assert len(self.bootstrapped_rewards) == self.buffer_size, \
            "Bootstrapped rewards have not been set yet. Cannot compute returns and advantages."

        self.returns, self.advantages = compute_returns_and_advantages(
            values=self.values,
            value_observers=self.value_observers,
            rewards=self.bootstrapped_rewards,
            reward_observers=self.reward_observers,  # rewards are observed by agents present in the next step
            dones=self.dones,
            device=self.device,
            actor_loc=self.actor_loc,
            value_loc=self.value_loc,
            gae_lambda=gae_lambda,
            discount_factor=discount_factor
        )

    def get_explained_variance(self) -> np.ndarray:
        LOG_CONSOLE_FUNCTION()

        if self.actor_loc == "multi" and self.value_loc == "single":
            values = [val.expand_as(ret) for ret, val in zip(self.returns, self.values)]
        else:
            values = self.values
        return explained_variance_function(torch.cat(values).detach().cpu().numpy(),
                                           torch.cat(self.returns).detach().cpu().numpy()
                                           )

    def _get_minibatch(self, batch_indices: np.ndarray, target_device: str) -> Minibatch:
        """
        Get samples from the buffer for the given indices.
        This is a helper function to be used in the get() method.
        """
        LOG_CONSOLE_FUNCTION()

        observations = [self.observations[i] for i in batch_indices]
        obs_agent_counts = [obs.get_agent_count(self.actor_loc) for obs in observations]
        obs_batch = ObservationBatch.from_obs_list(observations)
        actions_raw = torch.cat([self.actions_raw[i] for i in batch_indices])
        actions_sedi = torch.cat([self.actions_sedi[i] for i in batch_indices])
        logprobs = torch.cat([self.logprobs[i] for i in batch_indices])
        values = torch.cat([self.values[i] for i in batch_indices])
        advantages = torch.cat([self.advantages[i] for i in batch_indices])
        returns = torch.cat([self.returns[i] for i in batch_indices])

        return Minibatch(obs_batch.to(target_device).detach(),
                         torch.tensor(obs_agent_counts, dtype=torch.long, device=target_device),
                         actions_raw.to(target_device).detach(),
                         actions_sedi.to(target_device).detach(),
                         logprobs.to(target_device).detach(),
                         values.to(target_device).detach(),
                         advantages.to(target_device).detach(),
                         returns.to(target_device).detach()
                         )

    def minibatches(self, batch_size: int, target_device: str) -> Generator[Tuple[Minibatch, np.ndarray], None, None]:
        """
        Get a generator yielding samples from the buffer.
        """
        LOG_CONSOLE_FUNCTION()

        assert self.ready_for_training, "Buffer has not been properly filled, or advantages/returns not computed yet."
        indices = np.random.permutation(self.buffer_size)  # get random permutation of data

        # generate a piece of data for each call until all indices have been used.
        # Data is already detached and on target_device once it is put into the RolloutBufferSamples object.
        start_idx = 0
        while start_idx < self.buffer_size:
            batch_indices = indices[start_idx: start_idx + batch_size]
            yield self._get_minibatch(batch_indices, target_device), batch_indices
            start_idx += batch_size

    def vf_meta_learning_split(self,
                               ep_split_idx: Tensor,
                               step_split_idx: Tensor,
                               target_device: str) -> (Minibatch, ObservationBatch, Minibatch, ObservationBatch):
        """
        Get a two-part split of the buffer for value function meta-learning.
        """
        LOG_CONSOLE_FUNCTION()

        assert self.filled, "Buffer has not been properly filled."

        step_idx = np.arange(self.buffer_size)
        minibatch1 = self._get_minibatch(step_idx[step_split_idx.cpu().numpy() == 0], self.device)
        minibatch2 = self._get_minibatch(step_idx[step_split_idx.cpu().numpy() == 1], self.device)

        ep_idx = np.arange(self.num_episodes)
        terminal_obs1 = [self.terminal_observations[i] for i in ep_idx[ep_split_idx.cpu().numpy() == 0]]
        terminal_obs_batch1 = ObservationBatch.from_obs_list(terminal_obs1).to(target_device).detach()
        terminal_obs2 = [self.terminal_observations[i] for i in ep_idx[ep_split_idx.cpu().numpy() == 1]]
        terminal_obs_batch2 = ObservationBatch.from_obs_list(terminal_obs2).to(target_device).detach()

        return minibatch1, terminal_obs_batch1, minibatch2, terminal_obs_batch2
