from typing import List

import torch
from torch import Tensor

from src.utils.constants import BIRDSEYE_OBSERVER_ID  # global observer ID for the birdseye view


"""
We have put the return and advantage computation into a separate file to make it easier for us to test it 
(see test/test_ret_adv.py).
"""


def compute_returns_and_advantages(values: List[Tensor],
                                   value_observers: List[Tensor],
                                   rewards: List[Tensor],
                                   reward_observers: List[Tensor],
                                   dones: Tensor,
                                   device: torch.device,
                                   actor_loc: str,
                                   value_loc: str,
                                   gae_lambda: float,
                                   discount_factor: float,
                                   debug=False
                                   ) -> (List[Tensor], List[Tensor]):
    """
    Computes the lambda-return (TD(lambda) estimate) and GAE(lambda) advantage for the entire buffer.

    We may have multiple agents in the environment and a varying amount of them in between episodes.
    Moreover, we may not have rewards/value estimates for all agents at each step.
    Hence, we obtain the union of all observers in the buffer, create sparse tensors for rewards and values and
    compute the advantages and returns for the sparse tensors at each step before re-masking them.

    [GAE Adapted from Stable Baselines 3]
    Post-processing step: compute the lambda-return (TD(lambda) estimate) and GAE(lambda) advantage.

    Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438) to compute the advantage as
    GAE: h_t^V = r_t + y * V(s_t+1) - V(s_t)
    with
    V(s_t+1) = 0 if s_t is terminal (where V(s) may be bootstrapped in the algorithm sampling for terminal dones)
               V(s_t+1) if s_t not terminal
    where T is the last step of the rollout.

    To obtain vanilla advantage (A(s) = R - V(S))
    where R is the discounted reward with value bootstrap,
    set ``gae_lambda=1.0`` during initialization.

    The TD(lambda) estimator has also two special cases:
    - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
    - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

    For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

    Args:
        values: List of tensors containing the value estimates for each step in the buffer
        value_observers: List of tensors containing the observer IDs for each step in the buffer
        rewards: List of tensors containing the rewards for each step in the buffer.
        reward_observers: List of tensors containing the observer IDs for each step in the buffer
        dones: Tensor containing the done flag for each step in the buffer.
        device: Device to use for the computations (e.g., 'cpu' or 'cuda').
        actor_loc: Mode of the agent, either 'single' for single-agent or 'multi' for multi-agent.
        value_loc: Scope of the value function, either 'single' for centralized or 'multi' for decentralized.
        gae_lambda: Lambda parameter for GAE, typically between 0 and 1.
        discount_factor: Discount factor for the rewards, typically between 0 and 1.
        debug: If True, prints debug information.
    """
    if debug:
        print(f"actor_loc: {actor_loc}, "
              f"value_loc: {value_loc}, "
              f"gae_lambda: {gae_lambda}, "
              f"discount_factor: {discount_factor}")

    # 1. Get relevant observers from the buffer.
    # - 1a. Get all agents seen in the buffer for creating/using sparse tensors.
    all_observers = torch.cat(value_observers + reward_observers)

    if value_loc == "multi":
        relevant_value_observers = torch.unique(all_observers).tolist()
        relevant_value_observers.sort()
        if BIRDSEYE_OBSERVER_ID in relevant_value_observers:
            relevant_value_observers.remove(BIRDSEYE_OBSERVER_ID)  # remove the global observer ID if present
    else:
        relevant_value_observers = [BIRDSEYE_OBSERVER_ID]  # single value scope, birdseye observer

    if actor_loc == "multi":
        # Multi-agent: all non-birdseye observers seen at least once in the buffer.
        relevant_reward_observers = torch.unique(all_observers).tolist()
        relevant_reward_observers.sort()
        if BIRDSEYE_OBSERVER_ID in relevant_reward_observers:
            relevant_reward_observers.remove(BIRDSEYE_OBSERVER_ID)  # remove the global observer ID if present
    else:
        # Single-agent: birdseye observer as agent
        # (regardless of the actor_input, we always use the "global" reward as a training signal).
        relevant_reward_observers = [BIRDSEYE_OBSERVER_ID]

    if debug:
        print(f"relevant value observers: {relevant_value_observers}")
        print(f"relevant actor observers: {relevant_reward_observers}")
        print(f"values: {values}")
        print(f"rewards: {rewards}")

    # helpers
    obs_value_id_to_sparse_idx = {obs_id: idx for idx, obs_id in enumerate(relevant_value_observers)}
    obs_reward_id_to_sparse_idx = {obs_id: idx for idx, obs_id in enumerate(relevant_reward_observers)}
    N_value = len(relevant_value_observers)  # number of columns in the sparse tensors
    N_reward = len(relevant_reward_observers)  # number of columns in the sparse tensors
    T = len(dones)  # number of steps in the buffer, i.e., buffer_size
    not_dones = (1 - dones).unsqueeze(-1)  # 1 for not done, 0 for done

    if debug:
        print(f"obs_value_id_to_sparse_idx: {obs_value_id_to_sparse_idx}")
        print(f"obs_reward_id_to_sparse_idx: {obs_reward_id_to_sparse_idx}")

    # - 1b. Get relevant observers per step for buffer iteration
    relevant_value_observers_per_step = [
        sorted([i for i in value_observers[t].tolist() if i in relevant_value_observers])
        for t in range(T)
    ]
    relevant_reward_observers_per_step = [
        sorted([i for i in reward_observers[t].tolist() if i in relevant_reward_observers])
        for t in range(T)
    ]

    # - 1c. Create a mask for agents that are present in the NEXT step
    # (Deltas involve rewards, which are present for NEXT step's observers)
    agent_mask = torch.zeros((T, N_reward), device=device, dtype=torch.double)
    for t in range(T):
        for obs_id in relevant_reward_observers_per_step[t]:
            agent_mask[t, obs_reward_id_to_sparse_idx[obs_id]] = 1.0

    # 2. Pre-allocate sparse tensors for rewards and values, containing a column for each agent ID.
    # - # TODO support default fill values other than zero (e.g. last/avg)?
    values_sparse = torch.zeros((T, N_value), device=device, dtype=torch.double)
    rewards_sparse = torch.zeros((T, N_reward), device=device, dtype=torch.double)

    # 3. Fill the pre-allocated tensors with the actual rewards and values
    for t in range(T):  # +1 because we also need the last step's values

        # - 3a. fill values
        if value_loc == "single":
            # assign birdseye observer value to all agents (this applies either to the global agent, or the local ones).
            values_sparse[t, :] = values[t]
        else:
            # in other deployment modes, we have values for each agent in THIS step
            values_t = values[t]
            value_observers_t = value_observers[t].tolist()
            for i, obs_id in enumerate(relevant_value_observers_per_step[t]):
                value_t_i = values_t[value_observers_t.index(obs_id)]
                values_sparse[t, obs_value_id_to_sparse_idx[obs_id]] = value_t_i

        # - 3b. fill rewards, which are present for NEXT step's observers
        if t == T:
            continue  # last "step" consists of the final values only, no rewards
        if actor_loc == "single":
            # use birdseye observer reward
            rewards_sparse[t] = rewards[t]
        else:
            rewards_t = rewards[t]
            reward_observers_t = reward_observers[t].tolist()
            for i, obs_id in enumerate(relevant_reward_observers_per_step[t]):
                reward_t_i = rewards_t[reward_observers_t.index(obs_id)]
                rewards_sparse[t, obs_reward_id_to_sparse_idx[obs_id]] = reward_t_i

    # 3c. get next values
    next_values_sparse = torch.zeros((T, N_value), device=device, dtype=torch.double)
    next_values_sparse[:-1] = values_sparse[1:]
    # NOTE: during rollout we bootstrap terminal rewards as `reward += discount_factor * terminal_value`.
    #  therefore, we can safely keep the next_values entries for final timesteps as zero,
    #  because the bootstrapped terminal rewards make sure below delta computation still incorporates next values.

    # debug prints
    if debug:
        print(f"rewards_sparse: {rewards_sparse}")
        print(f"values_sparse: {values_sparse}")
        print(f"next_values_sparse: {next_values_sparse}")

    # 4. Compute deltas. Use a mask to set deltas to zero for agents that are not present in the next step.
    deltas_sparse = agent_mask * (rewards_sparse +
                                  discount_factor * next_values_sparse.expand_as(rewards_sparse) * not_dones -
                                  values_sparse.expand_as(rewards_sparse))

    if debug:
        print(f"deltas_sparse: {deltas_sparse}")
        print(f"last rewards_sparse: {rewards_sparse[-1]}")
        print(f"last values_sparse: {values_sparse[-1]}")
        print(f"last next_values_sparse: {next_values_sparse[-1]}")
        print(f"last deltas_sparse: {deltas_sparse[-1]}")

    # 5. Compute advantages by reverse accumulation.
    # Since we have set deltas to zero for agents that are not present in the next step,
    # We can safely use the last GAE as zero.
    last_gae = torch.zeros(N_reward, device=device, dtype=torch.double)
    advantages_sparse = torch.zeros_like(rewards_sparse, device=device, dtype=torch.double)
    for t_rev in reversed(range(T)):
        last_gae = deltas_sparse[t_rev] + discount_factor * gae_lambda * not_dones[t_rev] * last_gae
        advantages_sparse[t_rev] = last_gae

    # 6. Returns are computed as advantages + values.
    returns_sparse = advantages_sparse + values_sparse

    # 7. Retain only the relevant agents in the advantages and returns, and store into buffer.
    returns = []
    advantages = []
    for t in range(T):
        # We can only compute deltas (and thus advantages and returns) for agents that provide rewards in THIS step t,
        # i.e., agents that are present in the NEXT step t+1.
        # Note: If the agent present in the next step not in THIS step t, we can use a default value foe value_t.
        relevant_adv_observers_t = relevant_reward_observers_per_step[t]

        advantages_t = torch.stack([advantages_sparse[t, obs_reward_id_to_sparse_idx[id]]
                                    for id in relevant_adv_observers_t])
        advantages.append(advantages_t)

        returns_t = torch.stack([returns_sparse[t, obs_reward_id_to_sparse_idx[id]]
                                 for id in relevant_adv_observers_t])
        returns.append(returns_t)

    return returns, advantages
