from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
from omegaconf import DictConfig

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum

from .on_policy_buffer import OnPolicyBuffer, Minibatch
from .utils.losses import policy_loss as get_policy_loss, value_loss as get_value_loss
from .value_net import PPOValueNetwork
from src.algorithm.trainable import TrainableAlgorithm
from src.normalizer.reward import get_reward_normalizer
from src.packerl_env import PackerlEnv
from src.scenario import NetworkScenario
from src.trainer._trainer import Trainer
from src.utils.logging import LOG_CONSOLE_FUNCTION, LOG_CONSOLE_RUN, LOG_CONSOLE_INFO
from src.utils.constants import BIRDSEYE_OBSERVER_ID
from src.utils.observations import ObservationGraph, Observation, ObservationBatch
from src.utils.torch import train_stat_dict
from src.utils.time import TimedBlock
from src.utils.types import ActionRaw, ActionSedi, Reward, LogprobTensor, ValueTensor, InferenceTimeTensor
from src.utils.utils import list_of_dict_to_dict_of_lists, prefix_dict


def get_loc_from_ppo_flavor(ppo_flavor: str):
    # returns train_actor_loc, train_value_loc
    if ppo_flavor == "PPO":
        return "single", "single"
    elif ppo_flavor == "IPPO":
        return "multi", "single"
    elif ppo_flavor == "MAPPO":
        return "multi", "multi"
    elif ppo_flavor == "HVPPO":
        return "multi", "hybrid"
    else:
        raise ValueError(f"Unknown algorithm flavor/factorization: {ppo_flavor}")


def get_module_input(module_input: str, value_loc: str):
    # returns train_actor_input, train_value_input
    if module_input == "birdseye":
        return "birdseye", "birdseye"
    elif module_input == "central":
        return "central", "central"
    elif module_input == "local":
        if value_loc == "multi":
            train_value_input = "local"
        else:
            train_value_input = "birdseye"  # single-agent value function still uses global input
        return "local", train_value_input
    else:
        raise ValueError(f"Unknown module input type: {module_input}")


class PPOTrainer(Trainer):
    """
    PPO algorithm for both single-agent and multi-agent cases. Its algorithm name is determined in this module's
    init.py, depending on the deployment mode (single-agent or multi-agent, centralized/decentralized value function).
    """
    num_minibatches: int
    update_epochs: int
    discount_factor: float
    gae_lambda: float
    pi_lr: float
    pi_clip_range: float
    pi_max_kl: float
    pi_max_clip_fraction: float
    vf_lr: float
    vf_onehot_infuse: bool
    vf_pool: str
    vf_meta_learning: bool
    vf_meta_adapt_lr: float
    vf_scaling: float
    vf_clip_range: float
    ent_lr: float

    def __init__(self, config: DictConfig, trainer_config: DictConfig, env: PackerlEnv, algorithm: TrainableAlgorithm):
        """
        Initializes the algorithm by setting up the on-policy buffer, the actor-value module, and the normalizers.
        """
        super().__init__(config, trainer_config, env, algorithm)
        self._name = trainer_config.flavor
        self.train_actor_loc, self.train_value_loc = get_loc_from_ppo_flavor(trainer_config.flavor)
        self.train_actor_input, self.train_value_input = get_module_input(trainer_config.module_input,
                                                                          self.train_value_loc)

        if self.vf_meta_learning and self.num_event_seq_repeats < 2:
            self.num_event_seq_repeats = 2  # need at least 2 repeats to do a split for meta-learning
            self._train_scenario_generator.num_event_seq_repeats = 2  # update scenario generator accordingly
            LOG_CONSOLE_RUN(f"NOTE: num_event_seq_repeats increased to 2 to enable value function meta-learning.")

        episodes_per_rollout = (self.num_topologies
                                * self.num_event_seqs_per_topology
                                * self.num_event_seq_repeats)
        steps_per_rollout = episodes_per_rollout * self.ep_length
        if steps_per_rollout % self.num_minibatches != 0:
            raise ValueError(f"Number of steps per rollout ({steps_per_rollout})"
                             f" = rollout_ep_length ({self.ep_length})"
                             f" * num event seq repeats ({self.num_event_seq_repeats})"
                             f" * num event seqs per topology ({self.num_event_seqs_per_topology})"
                             f" * topologies per rollout ({self.num_topologies})"
                             f" must be divisible by number of minibatches ({self.num_minibatches})")
        self.minibatch_size = steps_per_rollout // self.num_minibatches
        if self.minibatch_size < 2:
            raise ValueError(f"Minibatch size ({self.minibatch_size}) must be at least 2")

        # do we meta-learn the value function?
        if self.vf_meta_learning:
            # create split tensor for meta-learning
            ep_lengths = torch.full((episodes_per_rollout,), self.ep_length, dtype=torch.long)
            vf_meta_ep_split_idx = torch.arange(episodes_per_rollout, dtype=torch.long) % 2
            vf_meta_step_split_idx = vf_meta_ep_split_idx.repeat_interleave(ep_lengths)
            self._train_step = lambda: self._train_step_vf_meta(vf_meta_ep_split_idx, vf_meta_step_split_idx)
        else:
            self._train_step = self._train_step_default

        # initialize on-policy buffer (stores the experience of the rollout phase, resets each iteration)
        self._buffer = OnPolicyBuffer(buffer_size=steps_per_rollout,
                                      num_episodes=episodes_per_rollout,
                                      actor_loc=self.train_actor_loc,
                                      value_loc=self.train_value_loc,
                                      device=self._algo.actor.device)

        # Initialize value function
        value_feat_counts = self._get_value_func_feat_counts(self._algo.feature_counts.copy())
        # in the 'hybrid' value scope, we concat global + local input for the value_net
        if self.train_value_loc == "hybrid":
            value_feat_counts = {k: v * 2 for k, v in value_feat_counts.items()}
        # If infusing one-hot scenario ID + timestep, adjust global feature counts accordingly
        self._vf_onehot_infuse_dim = 0
        if self.vf_onehot_infuse:
            if not self.fixed_scenarios:
                raise ValueError("PPO value function one-hot infusion is only supported with fixed scenarios.")
            self._vf_onehot_infuse_dim = (self._train_scenario_generator.num_topologies
                                          + self._train_scenario_generator.num_event_seqs_per_topology
                                          + self._train_scenario_generator.ep_length + 1  # +1 to cover t \in [0,T]
                                          )
        value_feat_counts["global"] += self._vf_onehot_infuse_dim
        self._value_function : nn.Module = PPOValueNetwork(self._algo.actor.nn_config,
                                                           feature_counts=value_feat_counts,
                                                           global_pool=self.vf_pool,
                                                           device=self._algo.actor.device)
        # print value function parameter count
        num_vf_params = sum(p.numel() for p in self._value_function.parameters() if p.requires_grad)
        LOG_CONSOLE_INFO(f"PPO Value Function initialized with {num_vf_params} trainable parameters. Input feature counts: {value_feat_counts}")

        # optimizers
        self.policy_optimizer = torch.optim.Adam(self._algo.actor_params, lr=self.pi_lr)
        self.value_optimizer = torch.optim.Adam(self._value_function.parameters(), lr=self.vf_lr)
        self._entropy_rho = torch.nn.Parameter(torch.tensor(0.0))
        self.entropy_optimizer = torch.optim.Adam([self._entropy_rho], lr=self.ent_lr)

        # we use an additional reward normalizer for PPO, which is reset at the beginning of each episode
        self._reward_normalizer = get_reward_normalizer(trainer_config.reward.normalize, self.discount_factor)
        self.lmb_reward = trainer_config.reward.global_to_local_reward_ratio

        # we start in eval mode
        self._set_train(False)

    def _set_train(self, train: bool):
        """
        Sets the algorithm and this trainer's value function to training or evaluation mode
        """
        self._algo.set_train(train)
        if train:
            self._value_function.train()
        else:
            self._value_function.eval()

    def __str__(self):
        return self._name

    def _entropy_alpha(self) -> torch.Tensor:
        # alpha used in policy loss
        return F.softplus(self._entropy_rho)

    def _get_value_func_feat_counts(self, feat_counts: dict):
        """
        Returns the value feature counts for the value function, depending on the
        specific PPO trainer implementation. Implemented in subclasses.
        """
        raise NotImplementedError

    def _get_value_view(self, obs: Observation | ObservationBatch) -> Observation | ObservationBatch:
        raise NotImplementedError

    def _get_value(self,
                   obs: Observation | ObservationBatch,
                   value_function: nn.Module,
                   grad: bool = False,
                   get_detached_list: bool = False) -> (ValueTensor, List[ValueTensor] | None):
        """
        Computes the value estimates for a given observation or observation batch, using the provided value function.
        Args:
            obs: Observation or ObservationBatch to compute the value for.
            value_function: The value function module to use for computation.
            grad: Whether to compute gradients (True) or not (False).
            get_detached_list: Whether to return a list of detached value tensors per step (e.g. for buffer storage).
        """

        obs_view = self._get_value_view(obs).clone().to(value_function.device)
        if self._vf_onehot_infuse_dim > 0:
            if not hasattr(obs_view, "s_t_onehot"):
                raise ValueError("Observation does not have 's_t_onehot' attribute for vf one-hot infusion.")
            if isinstance(obs_view, ObservationBatch):
                infused_s_t = obs_view.s_t_onehot.repeat_interleave(obs_view.num_graphs_per_observation, dim=0)
            else:
                infused_s_t = obs_view.s_t_onehot.expand(obs_view.u.shape[0], -1)
            obs_view.__setattr__("u", torch.cat([obs_view.u, infused_s_t], dim=-1))
        if grad:
            value = value_function(obs_view).squeeze(dim=-1)
        else:
            with torch.no_grad():
                value = value_function(obs_view).squeeze(dim=-1)

        if get_detached_list:
            value_list = value.detach().split(obs_view.num_graphs_per_observation.tolist())
        else:
            value_list = None
        return value, value_list

    # ------------------- RL Specific algorithm functions -------------------

    def _reset_env(self, scenario: NetworkScenario):
        obs = self._algo.reset_env(scenario)
        self._reward_normalizer.reset(None)
        return obs

    def _sample_action(self, obs_graph: ObservationGraph) -> (ActionRaw, ActionSedi, LogprobTensor):
        raise NotImplementedError

    def get_rollout_action(self, obs: Observation) \
            -> (ActionRaw, ActionSedi, LogprobTensor, ValueTensor, InferenceTimeTensor):
        """
        The PPO algorithm uses the actor-value module to get a sampled action during rollout.
        """
        LOG_CONSOLE_FUNCTION()

        obs_view_graphs = self._algo.get_actor_view(obs).to_obs_graph_list()
        action_raw_list, action_sedi_list, logprob_list, inference_times_ms = [], [], [], []
        for obs_graph in obs_view_graphs:
            with torch.no_grad():
                with TimedBlock() as inference_timer:
                    action_raw, action_sedi, logprob = self._sample_action(obs_graph)
            action_raw_list.append(action_raw)
            action_sedi_list.append(action_sedi)
            logprob_list.append(logprob)
            inference_times_ms.append(round(inference_timer.elapsed_ms, 3))

        value, _ = self._get_value(obs, self._value_function, grad=False)

        return (torch.cat(action_raw_list),
                torch.cat(action_sedi_list),
                torch.cat(logprob_list),
                value,
                torch.tensor(inference_times_ms, dtype=torch.float32)
                )

    def _blend_and_filter_rewards(self, reward: Reward) -> Reward:
        """
        Blends the global and local rewards of an observation according to the blending hyperparameter.
        Then, returns those blended rewards that are relevant for the agent mode
        """
        LOG_CONSOLE_FUNCTION()

        reward_values, reward_observers = reward
        is_global = reward_observers == BIRDSEYE_OBSERVER_ID
        local_rewards = reward_values[~is_global]
        mean_global_reward = reward_values[is_global] / local_rewards.numel()

        if self.train_actor_loc == "single":  # single-agent training deployment: one blended reward
            local_component = (1 - self.lmb_reward) * local_rewards.mean()
            global_component = self.lmb_reward * mean_global_reward
            blended_reward_observers = reward_observers[is_global]
        else:  # multi-agent training deployment: blended reward per agent
            local_component = (1 - self.lmb_reward) * local_rewards
            global_component = self.lmb_reward * mean_global_reward.expand_as(local_rewards)
            blended_reward_observers = reward_observers[~is_global]
        blended_reward = local_component + global_component
        return blended_reward, blended_reward_observers

    def _post_rollout_step(self,
                           obs: Observation,
                           action_raw: ActionRaw,
                           action_sedi: ActionSedi,
                           logprob: LogprobTensor,
                           values: ValueTensor,
                           reward: Reward,
                           next_obs_raw: Observation,
                           done: bool,
                           infos: Dict[str, Any]) -> Observation:
        """
        During Rollout, PPO normalizes the next observation and reward, blends the rewards,
        and stores the transition in the buffer.
        """
        LOG_CONSOLE_FUNCTION()

        # normalization
        next_obs = self._algo._prepare_obs(next_obs_raw)
        next_obs = self._algo.obs_normalizer.update_and_normalize(next_obs)
        reward = self._reward_normalizer.update_and_normalize(reward)
        normalized_rewards, reward_observers = reward
        infos["global_reward_normalized"] = normalized_rewards[reward_observers == BIRDSEYE_OBSERVER_ID].item()
        infos["local_reward_normalized"] = normalized_rewards[reward_observers != BIRDSEYE_OBSERVER_ID].tolist()

        # blend and filter rewards by relevant observers
        blended_rewards, blended_reward_observers = self._blend_and_filter_rewards(reward)
        infos["blended_reward"] = blended_rewards.tolist()

        # Since sedi are obtained per agent/observation graph, but in PPO we're evaluating batched actions,
        # we need to offset the sedi indices accordingly.
        action_sedi_with_offset = self._algo.get_action_sedi_offsets(self._algo.get_actor_view(obs))

        # values are computed just on the relevant observers, but when storing them into the buffer we don't
        # know which observers have been deemed relevant at that time because we store the full observation.
        # Therefore, we compute the relevant value observers here and store them in the buffer along with the values.
        value_observers = self._get_value_view(obs).observer_ids

        # store transition in buffer (full observations, non-masked tensors for action, logprob, value, reward)
        self._buffer.add(obs=obs.clone(),  # clone() because we assign next_obs to obs later on
                         action_raw=action_raw,
                         action_sedi=action_sedi_with_offset,
                         logprob=logprob,
                         value=(values, value_observers),
                         reward=(blended_rewards, blended_reward_observers),
                         done=Tensor([float(done)]),
                         terminal_obs=next_obs if done else None
                         )

        return next_obs

    # ====== Training ============================================================

    def _evaluate_action(self,
                         rollout_obs_batch_view: ObservationBatch,
                         rollout_actions_raw: ActionRaw,
                         rollout_actions_sedi: ActionSedi):
        raise NotImplementedError

    def _get_logprob_mask(self, rollout_obs_batch_view: ObservationBatch) -> Tensor:
        """
        Returns a mask for the logprobs that filters the values that should be considered in NN parameter updates.
        For single-agent/global agent cases, this is true wherever the observer ID belonging to the logprob entry
        matches GLOBAL_OBSERVER_ID/central_node ID (depending on agent input type), and for multi-agent cases,
        it depends on the policy design:
        - for next-hop policies, this is true wherever observer_ids == agent_ids
        - for link-weight policies, this is true wherever observer_ids != GLOBAL_OBSERVER_ID
        """
        raise NotImplementedError

    def _get_logprob_graph_idx(self, rollout_obs_batch_view: ObservationBatch) -> Tensor:
        raise NotImplementedError

    def _get_entropy_mask(self, rollout_obs_batch_view: ObservationBatch) -> Tensor:
        raise NotImplementedError

    def _update_vf_from_minibatch(self,
                                  minibatch: Minibatch,
                                  value_function: nn.Module,
                                  optimizer: torch.optim.Optimizer):
        """
        Update the value function for a given minibatch.
        """
        LOG_CONSOLE_FUNCTION()

        # unpack minibatch
        rollout_obs_batch, rollout_obs_agent_counts, _, _, _, rollout_values, _, returns = minibatch

        # obtain the training values for the observations in the rollout batch
        with TimedBlock() as get_value_timer:
            train_values, _ = self._get_value(rollout_obs_batch, value_function, grad=True)

        # calculate losses (all done with double precision).
        with TimedBlock() as loss_timer:
            # for MAPPO's value loss computation, expand training/rollout values to match num of agents in the return
            if self.train_actor_loc == "multi" and self.train_value_loc == "single":
                train_values = train_values.repeat_interleave(rollout_obs_agent_counts)
                rollout_values = rollout_values.repeat_interleave(rollout_obs_agent_counts)
            value_loss = get_value_loss(returns, train_values.double(), rollout_values.double(), self.vf_clip_range)

        # OPTIMIZATION STEP
        with TimedBlock() as opt_step_timer:
            vf_loss_term = self.vf_scaling * value_loss
            unclipped_vf_grads, clipped_vf_grads = self._opt_step(optimizer, vf_loss_term)

        # --- Logging ---
        # compute value clip function
        with torch.no_grad():
            value_outliers = torch.abs(train_values.detach() - rollout_values.detach()) > self.vf_clip_range
            value_clip_fraction = value_outliers.float().mean()

        train_step_scalars = {
            "value_loss": value_loss.item(),
            "value_clip_fraction": value_clip_fraction.item(),
        }
        train_step_scalars |= train_stat_dict(train_values, "value")
        train_step_scalars |= train_stat_dict(unclipped_vf_grads, "unclipped_vf_gradient")
        train_step_scalars |= train_stat_dict(clipped_vf_grads, "clipped_vf_gradient")

        times_s = {
            "get_train_values_time_s": round(get_value_timer.elapsed_s, 3),
            "vf_loss_calc_time_s": round(loss_timer.elapsed_s, 3),
            "vf_opt_step_time_s": round(opt_step_timer.elapsed_s, 3),
        }

        return train_step_scalars, times_s

    def _update_pi_from_minibatch(self, minibatch: Minibatch):
        """
        Update the policy from a given minibatch.
        """
        LOG_CONSOLE_FUNCTION()

        # unpack minibatch
        (rollout_obs_batch,
         _,
         rollout_actions_raw,
         rollout_actions_sedi,
         rollout_logprobs,
         _,
         advantages,
         _) = minibatch

        # evaluate the actions taken during the rollout phase
        rollout_obs_batch_view = self._algo.get_actor_view(rollout_obs_batch)
        with TimedBlock() as eval_actions_timer:
            train_logprobs, entropies, target_entropies = self._evaluate_action(rollout_obs_batch_view,
                                                                                rollout_actions_raw,
                                                                                rollout_actions_sedi)

        # Get logprobs, ratios from logprobs (pre-clamping to avoid numerical instability in the exp() operation),
        # normalized advantages, and relevant entropies.
        # Logprobs of training and rollout should be the same values at first epoch.
        # We need to mask the logprobs to only include the values that are actually relevant for the ratio calculation
        # (e.g. filtering out logprob components of agent actions that fall outside the agent's sphere of influence).
        with TimedBlock() as prep_ratios_adv_entropy_timer:
            logprob_mask = self._get_logprob_mask(rollout_obs_batch_view)
            ratio_idx = self._get_logprob_graph_idx(rollout_obs_batch_view)
            entropy_mask = self._get_entropy_mask(rollout_obs_batch_view)

            train_logprobs = train_logprobs[logprob_mask]
            rollout_logprobs = rollout_logprobs[logprob_mask]
            # obtain the ratios of the logprobs, then average them per graph (i.e. per step and agent) via ratio_idx
            ratios_raw = torch.exp(torch.clamp(train_logprobs - rollout_logprobs, min=-2.0, max=2.0))
            ratios = scatter_mean(ratios_raw, ratio_idx[logprob_mask], dim=0)
            normalized_advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8))
            entropies = entropies[entropy_mask]
            if target_entropies is not None:
                target_entropies = target_entropies[entropy_mask]

        # calculate losses (all done with double precision).
        with TimedBlock() as loss_timer:
            policy_loss = get_policy_loss(normalized_advantages, ratios, self.pi_clip_range)
            entropy_loss = -torch.mean(entropies)
            entropy_alpha = self._entropy_alpha().to(entropies.device)
            if target_entropies is not None:
                exploration_loss = entropy_alpha * (entropies.mean().detach() - target_entropies.mean().detach())
            else:
                exploration_loss = None  # no exploration loss if no target entropy is provided

        # OPTIMIZATION STEP
        with TimedBlock() as opt_step_timer:
            pi_loss_term = policy_loss + entropy_alpha.detach() * torch.mean(entropies)
            unclipped_pi_grads, clipped_pi_grads = self._opt_step(self.policy_optimizer, pi_loss_term)
            if exploration_loss is not None:
                unclipped_en_grads, clipped_en_grads = self._opt_step(self.entropy_optimizer, exploration_loss)

        # --- Logging ---
        # compute approximate KL divergence, clip fractions, and normalized entropy
        with torch.no_grad():
            log_ratio = train_logprobs - rollout_logprobs.detach()
            lr_exp = (torch.exp(log_ratio) - 1) - log_ratio
            policy_kl = torch.mean(scatter_sum(lr_exp, ratio_idx[logprob_mask], dim=0))
            normalized_policy_kl = torch.mean(scatter_mean(lr_exp, ratio_idx[logprob_mask], dim=0))
            ratio_outliers = torch.abs(ratios.detach() - 1) > self.pi_clip_range
            policy_clip_fraction = ratio_outliers.float().mean()
            if target_entropies is not None:
                normalized_entropy = (entropies / target_entropies.clamp_min(1e-7)).mean()  # in [0,1]
            else:
                normalized_entropy = entropies.mean()

        train_step_scalars = {
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "policy_kl": policy_kl.item(),
            "normalized_policy_kl": normalized_policy_kl.item(),
            "policy_clip_fraction": policy_clip_fraction.item(),
            "normalized_entropy": normalized_entropy.item(),
        }
        train_step_scalars |= train_stat_dict(train_logprobs, "logprob")
        train_step_scalars |= train_stat_dict(rollout_logprobs, "old_logprob")
        train_step_scalars |= train_stat_dict(advantages, "adv")
        train_step_scalars |= train_stat_dict(normalized_advantages, "normalized_adv")
        train_step_scalars |= train_stat_dict(ratios, "ratios")
        train_step_scalars |= train_stat_dict(entropies, "entropy")
        train_step_scalars |= train_stat_dict(unclipped_pi_grads, "unclipped_pi_gradient")
        train_step_scalars["unclipped_pi_gradient"] = unclipped_pi_grads.detach()
        train_step_scalars |= train_stat_dict(clipped_pi_grads, "clipped_pi_gradient")
        train_step_scalars["clipped_pi_gradient"] = clipped_pi_grads.detach()
        if exploration_loss is not None:
            train_step_scalars |= train_stat_dict(unclipped_en_grads, "unclipped_en_gradient")
            train_step_scalars |= train_stat_dict(clipped_en_grads, "clipped_en_gradient")
        if target_entropies is not None:
            train_step_scalars |= train_stat_dict(target_entropies, "target_entropy")

        times_s = {
            "evaluate_actions_time_s": round(eval_actions_timer.elapsed_s, 3),
            "ratio_adv_entropy_time_s": round(prep_ratios_adv_entropy_timer.elapsed_s, 3),
            "pi_loss_calc_time_s": round(loss_timer.elapsed_s, 3),
            "pi_opt_step_time_s": round(opt_step_timer.elapsed_s, 3),
        }

        return train_step_scalars, times_s

    # ======= Training Step ============================================================

    def _train_step_default(self):
        """
        A training step consists of multiple training epochs on the entire rollout 'buffer'
        (i.e. the rollout phase's experience), where in turn an epoch executes multiple update steps on a minibatch.
        The number of epochs is specified in the config.
        """
        LOG_CONSOLE_FUNCTION()
        LOG_CONSOLE_RUN(f"--- training step ({self.update_epochs} update epochs, default) ---")

        old_actor_params = torch.cat([param.detach().flatten() for param in self._algo.actor_params])
        old_value_params = torch.cat([param.detach().flatten() for param in self._value_function.parameters()])

        # preparation
        assert self._buffer.filled, "Cannot perform training step because the on-policy buffer is not full!"
        terminal_obs = ObservationBatch.from_obs_list(self._buffer.terminal_observations)
        _, terminal_values = self._get_value(terminal_obs, self._value_function, grad=False, get_detached_list=True)
        self._buffer.bootstrap_rewards(terminal_values, self.discount_factor)
        self._buffer.compute_returns_and_advantages(self.gae_lambda, self.discount_factor)

        update_infos = []
        unclipped_pi_grads = []
        clipped_pi_grads = []
        update_times = []

        # --- training core ---
        pi_active = True

        for e in range(self.update_epochs):
            update_infos_epoch = []
            update_times_epoch = []

            # Do a complete pass on the rollout buffer, sampling random minibatches.
            for minibatch, _ in self._buffer.minibatches(self.minibatch_size, self._algo.actor.device):
                if pi_active:
                    pi_update_info, pi_update_times_s = self._update_pi_from_minibatch(minibatch)
                    unclipped_pi_grads.append(pi_update_info.pop("unclipped_pi_gradient"))
                    clipped_pi_grads.append(pi_update_info.pop("clipped_pi_gradient"))
                else:
                    pi_update_info, pi_update_times_s = {}, {}  # no pi update this epoch
                vf_update_info, vf_update_times_s = self._update_vf_from_minibatch(minibatch,
                                                                                   self._value_function,
                                                                                   self.value_optimizer)
                update_infos_epoch.append(pi_update_info | vf_update_info)
                update_times_epoch.append(pi_update_times_s | vf_update_times_s)

            # check for early stopping based on KL divergence or policy clip fraction (at least 1 epoch completed)
            if "normalized_policy_kl" in update_infos_epoch[0]:
                mean_kl = np.mean([info["normalized_policy_kl"] for info in update_infos_epoch])
            else:
                mean_kl = None
            if pi_active and 0.0 < self.pi_max_kl < mean_kl:
                pi_active = False
                LOG_CONSOLE_INFO(f"Early stopping of policy updates at epoch {e} "
                                 f"due to high KL divergence: {mean_kl.round(3)} > {self.pi_max_kl}")

            if "policy_clip_fraction" in update_infos_epoch[0]:
                mean_clip_frac = np.mean([info["policy_clip_fraction"] for info in update_infos_epoch])
            else:
                mean_clip_frac = None
            if pi_active and 0.0 < self.pi_max_clip_fraction < mean_clip_frac:
                pi_active = False
                LOG_CONSOLE_INFO(f"Early stopping of policy updates at epoch {e} " 
                                 f"due to high clip fraction: {mean_clip_frac.round(3)} > {self.pi_max_clip_fraction}")

            update_infos.extend(update_infos_epoch)
            update_times.extend(update_times_epoch)
            LOG_CONSOLE_INFO(f"update epoch {e}: "
                             f"mean normalized KL = {mean_kl.round(3) if mean_kl is not None else None}, "
                             f"mean clip fraction = {mean_clip_frac.round(3) if mean_clip_frac is not None else None}, "
                             f"pi_active = {pi_active}")
        # --- training core end ---

        # logging
        train_scalars = {}

        # - pi/ent/vf train scalars
        update_infos = list_of_dict_to_dict_of_lists(update_infos)
        for k, v in update_infos.items():
            train_scalars |= train_stat_dict(v, k)
        train_scalars["unclipped_pi_grad_variance"] = torch.stack(unclipped_pi_grads, dim=0).var(dim=0).mean().item()
        train_scalars["clipped_pi_grad_variance"] = torch.stack(clipped_pi_grads, dim=0).var(dim=0).mean().item()
        new_actor_params = torch.cat([param.detach().flatten() for param in self._algo.actor_params])
        actor_param_diffs = torch.abs(old_actor_params - new_actor_params)
        train_scalars["mean_actor_weight_difference"] = actor_param_diffs.mean().item()
        train_scalars["max_actor_weight_difference"] = actor_param_diffs.max().item()
        new_value_params = torch.cat([param.detach().flatten() for param in self._value_function.parameters()])
        value_param_diffs = torch.abs(old_value_params - new_value_params)
        train_scalars["mean_value_weight_difference"] = value_param_diffs.mean().item()
        train_scalars["max_value_weight_difference"] = value_param_diffs.max().item()
        train_scalars["value_explained_variance"] = self._buffer.get_explained_variance()
        train_scalars["entropy_rho"] = self._entropy_rho.item()
        if hasattr(self, "epsilon"):  # include epsilon if implemented in epsilon-greedy subclass
            train_scalars["epsilon"] = self.epsilon

        # - training times
        update_times = list_of_dict_to_dict_of_lists(update_times)
        train_scalars |= {f"times/{k}": round(np.sum(v), 3) for k, v in update_times.items()}  # training times

        return train_scalars

    def _train_step_vf_meta(self, vf_meta_ep_split_idx: Tensor, vf_meta_step_split_idx: Tensor):
        """
        A training step with value-function meta-learning as per Mao et al.,
        "Variance Reduction for Reinforcement Learning in Input-Driven Environments" (ICLR 2019),
        https://openreview.net/pdf?id=Hyg1G2AqtQ. The algorithm's steps:
        A. Split the rollout buffer into two halves and create two adaptations of the value function on each half.
        B. Use the adapted value functions to obtain baseline value estimates for the opposite half of the buffer.
        D. Update the policy using the adapted baseline estimates as value estimates for advantage/return computation.
        C. Compute the meta value loss from the baseline estimates and do a MAML-style meta-optimization step.

        Note that we perform the meta value optimization step BEFORE the policy update step, unlike the original paper,
        so that we can reduce memory usage by overriding the buffer values with the adapted ones for the pi update step.

        Args:
            vf_meta_ep_split_idx: Tensor of shape (num_episodes,) with 0/1 entries indicating the split assignment
                of each episode in the buffer.
            vf_meta_step_split_idx: Tensor of shape (buffer_size,) with 0/1 entries indicating the split assignment
                of each step in the buffer.
        """
        LOG_CONSOLE_FUNCTION()
        LOG_CONSOLE_RUN(f"--- training step ({self.update_epochs} update epochs, VF-Meta) ---")

        old_actor_params = torch.cat([param.detach().flatten() for param in self._algo.actor_params])
        old_value_params = torch.cat([param.detach().flatten() for param in self._value_function.parameters()])

        # preparation
        assert self._buffer.filled, "Cannot perform training step because the on-policy buffer is not full!"
        terminal_obs = ObservationBatch.from_obs_list(self._buffer.terminal_observations)
        _, vf_terminals_list = self._get_value(terminal_obs, self._value_function, grad=False, get_detached_list=True)
        self._buffer.bootstrap_rewards(vf_terminals_list, self.discount_factor)
        self._buffer.compute_returns_and_advantages(self.gae_lambda, self.discount_factor)

        # --- training core ---
        # 1. create 2 adaptations of the value func on the two splits, SGD-update them, get adapted value estimates.
        # 1a. split buffer into two halves
        buf_s1, terminal_obs_s1, buf_s2, terminal_obs_s2 \
            = self._buffer.vf_meta_learning_split(vf_meta_ep_split_idx, vf_meta_step_split_idx, self._algo.actor.device)

        # 1b. adapt vf1 on split 1
        vf1 = deepcopy(self._value_function).to(self._value_function.device)
        vf1_optimizer = torch.optim.SGD(vf1.parameters(), lr=self.vf_meta_adapt_lr)
        vf1_update_info, vf1_train_times_s = self._update_vf_from_minibatch(buf_s1, vf1, vf1_optimizer)

        # 1c. get vf1-adapted values for split 2
        with TimedBlock() as get_adapted_values_vf1_timer:
            vf1_vals_s2, vf1_vals_s2_list = self._get_value(buf_s2.observations, vf1, grad=True, get_detached_list=True)
            _, vf1_terminals_s2_list = self._get_value(terminal_obs_s2, vf1, grad=False, get_detached_list=True)
        vf1_train_times_s["get_adapted_values_time_s"] = round(get_adapted_values_vf1_timer.elapsed_s, 3)

        # 1d. adapt vf2 on split 2
        vf2 = deepcopy(self._value_function).to(self._value_function.device)
        vf2_optimizer = torch.optim.SGD(vf2.parameters(), lr=self.vf_meta_adapt_lr)
        vf2_update_info, vf2_train_times_s = self._update_vf_from_minibatch(buf_s2, vf2, vf2_optimizer)

        # 1e. get vf2-adapted values for split 1
        with TimedBlock() as get_adapted_values_vf2_timer:
            vf2_vals_s1, vf2_vals_s1_list = self._get_value(buf_s1.observations, vf2, grad=True, get_detached_list=True)
            _, vf2_terminals_s1_list = self._get_value(terminal_obs_s1, vf2, grad=False, get_detached_list=True)
        vf2_train_times_s["get_adapted_values_time_s"] = round(get_adapted_values_vf2_timer.elapsed_s, 3)

        # 2. calculate parts of the meta value loss (from adapted cross-baselines) and do vf-meta optimization step.
        # NOTE 1: we implement a first-order MAML, i.e., we DO NOT backpropagate through the adaptation steps
        # but treat the adapted value estimates as constant targets. This simplifies the logic and reduces memory usage.
        # See original MAML paper (Finn et al., ICML 2017) for details: https://arxiv.org/pdf/1703.03400.pdf
        # NOTE 2: Mao's meta input-dependent baseline algorithm does the meta vf update AFTER the policy update,
        # but we do it BEFORE the policy update so that we can simply override the buffer values with the
        # adapted values for the policy update step (avoids cloning the other buffer data).
        with TimedBlock() as meta_vf_loss_timer:
            vf1_loss = get_value_loss(returns=buf_s2.returns, train_values=vf1_vals_s2.double(),
                                      rollout_values=buf_s2.old_values.double(), clip_range=self.vf_clip_range)
            vf2_loss = get_value_loss(returns=buf_s1.returns, train_values=vf2_vals_s1.double(),
                                      rollout_values=buf_s1.old_values.double(), clip_range=self.vf_clip_range)
        with TimedBlock() as meta_vf_opt_step_timer:
            # Compute gradients wrt. adapted value functions, and copy these back onto the meta value function.
            # We need to set allow_unused=True because some parts of the value network are not used in the forward pass.
            # TODO re-design value net to remove global module in last MPN step?
            grads1 = torch.autograd.grad(self.vf_scaling * vf1_loss, vf1.parameters(),
                                         retain_graph=True, allow_unused=True)
            grads2 = torch.autograd.grad(self.vf_scaling * vf2_loss, vf2.parameters(),
                                         allow_unused=True)
            self.value_optimizer.zero_grad()
            with torch.no_grad():
                for p_meta, g1, g2 in zip(self._value_function.parameters(), grads1, grads2):
                    g_total = torch.zeros_like(p_meta)
                    if g1 is not None:
                        g_total += g1
                    if g2 is not None:
                        g_total += g2
                    p_meta.grad = g_total
            unclipped_meta_vf_grads = self.get_optimizer_flat_grads(self.value_optimizer)
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.get_optimizer_param_list(self.value_optimizer), self.max_grad_norm)
            clipped_meta_vf_grads = self.get_optimizer_flat_grads(self.value_optimizer)
            self.value_optimizer.step()

        # 3. adapt advantages by replacing value estimates with adapted cross-baselines.
        #  Adapted value lists have been detached to avoid interfering with the meta-update computation graph.
        def merge_by_split_idx(split_idx: Tensor, list1: List, list2: List) -> List:
            iters = (iter(list1), iter(list2))
            return [next(iters[idx]) for idx in split_idx.tolist()]
        adapted_baselines = merge_by_split_idx(vf_meta_step_split_idx, vf2_vals_s1_list, vf1_vals_s2_list)
        for i in range(self._buffer.buffer_size):
            self._buffer.advantages[i] += self._buffer.values[i] - adapted_baselines[i]

        # 4. update policy/entropy using the entire buffer with adapted values/returns/advantages.
        pi_update_infos = []
        unclipped_pi_grads = []
        clipped_pi_grads = []
        pi_update_times_s = []

        pi_active = True

        for e in range(self.update_epochs):

            pi_update_infos_epoch = []
            pi_update_times_epoch = []

            # Do a complete pass on the rollout buffer, sampling random minibatches.
            for minibatch, _ in self._buffer.minibatches(self.minibatch_size, self._algo.actor.device):
                if pi_active:
                    pi_upd_info, pi_upd_times_s = self._update_pi_from_minibatch(minibatch)
                    unclipped_pi_grads.append(pi_upd_info.pop("unclipped_pi_gradient"))
                    clipped_pi_grads.append(pi_upd_info.pop("clipped_pi_gradient"))
                else:
                    pi_upd_info, pi_upd_times_s = {}, {}  # no pi update this epoch
                pi_update_infos_epoch.append(pi_upd_info)
                pi_update_times_epoch.append(pi_upd_times_s)

            if "normalized_policy_kl" in pi_update_infos_epoch[0]:
                mean_kl = np.mean([info["normalized_policy_kl"] for info in pi_update_infos_epoch])
            else:
                mean_kl = None
            if pi_active and 0.0 < self.pi_max_kl < mean_kl:
                pi_active = False
                LOG_CONSOLE_INFO(f"Early stopping of policy updates at epoch {e} "
                                 f"due to high KL divergence: {mean_kl.round(3)} > {self.pi_max_kl}")
            if "policy_clip_fraction" in pi_update_infos_epoch[0]:
                mean_clip_frac = np.mean([info["policy_clip_fraction"] for info in pi_update_infos_epoch])
            else:
                mean_clip_frac = None
            if pi_active and 0.0 < self.pi_max_clip_fraction < mean_clip_frac:
                pi_active = False
                LOG_CONSOLE_INFO(f"Early stopping of policy updates at epoch {e} " 
                                 f"due to high clip fraction: {mean_clip_frac.round(3)} > {self.pi_max_clip_fraction}")

            LOG_CONSOLE_INFO(f"update epoch {e}: "
                             f"mean normalized KL = {mean_kl.round(3) if mean_kl is not None else None}, "
                             f"mean clip fraction = {mean_clip_frac.round(3) if mean_clip_frac is not None else None}, "
                             f"pi_active = {pi_active}")

        # --- training core end ---

        # logging (collect various stats/timer times/gradient info etc.)
        train_scalars = {}

        # - vf (vf1, vf2, meta) train scalars
        with torch.no_grad():
            value_outliers1 = torch.abs(vf1_vals_s2.detach() - buf_s2.old_values.detach()) > self.vf_clip_range
            value_outliers2 = torch.abs(vf2_vals_s1.detach() - buf_s1.old_values.detach()) > self.vf_clip_range
            vn1, vn2 = value_outliers1.numel(), value_outliers2.numel()
            value_clip_fraction = (value_outliers1.float().sum() + value_outliers2.float().sum()) / (vn1 + vn2)
        vf_meta_update_info = {
            "value_loss": (vf1_loss.detach() + vf2_loss.detach()).item(),
            "value_clip_fraction": value_clip_fraction.item(),
        }
        vf_meta_update_info |= train_stat_dict(unclipped_meta_vf_grads, "unclipped_vf_gradient")
        vf_meta_update_info |= train_stat_dict(clipped_meta_vf_grads, "clipped_vf_gradient")
        vf_meta_times_s = {
            "vf_loss_calc_time_s": round(meta_vf_loss_timer.elapsed_s, 3),
            "vf_opt_step_time_s": round(meta_vf_opt_step_timer.elapsed_s, 3),
        }
        new_value_params = torch.cat([param.detach().flatten() for param in self._value_function.parameters()])
        value_param_diffs = torch.abs(old_value_params - new_value_params)
        train_scalars["mean_value_weight_difference"] = value_param_diffs.mean().item()
        train_scalars["max_value_weight_difference"] = value_param_diffs.max().item()
        train_scalars["value_explained_variance"] = self._buffer.get_explained_variance()
        train_scalars |= {f"vf1_{k}": v for k, v in vf1_update_info.items()}
        train_scalars |= {f"vf2_{k}": v for k, v in vf2_update_info.items()}
        train_scalars |= {f"vfm_{k}": v for k, v in vf_meta_update_info.items()}

        # - pi/ent train scalars
        pi_update_infos = list_of_dict_to_dict_of_lists(pi_update_infos)
        for k, v in pi_update_infos.items():
            train_scalars |= train_stat_dict(v, k)
        train_scalars["unclipped_pi_grad_variance"] = torch.stack(unclipped_pi_grads, dim=0).var(dim=0).mean().item()
        train_scalars["clipped_pi_grad_variance"] = torch.stack(clipped_pi_grads, dim=0).var(dim=0).mean().item()
        new_actor_params = torch.cat([param.detach().flatten() for param in self._algo.actor_params])
        actor_param_diffs = torch.abs(old_actor_params - new_actor_params)
        train_scalars["mean_actor_weight_difference"] = actor_param_diffs.mean().item()
        train_scalars["max_actor_weight_difference"] = actor_param_diffs.max().item()
        train_scalars["entropy_rho"] = self._entropy_rho.item()
        if hasattr(self, "epsilon"):  # include epsilon if implemented in epsilon-greedy subclass
            train_scalars["epsilon"] = self.epsilon

        # - training times
        pi_update_times_s = list_of_dict_to_dict_of_lists(pi_update_times_s)
        train_scalars |= {f"times/{k}": round(np.sum(v), 3) for k, v in pi_update_times_s.items()}  # pi update
        train_scalars |= {f"times/vf1_{k}": v for k, v in vf1_train_times_s.items()}  # vf1 update
        train_scalars |= {f"times/vf2_{k}": v for k, v in vf2_train_times_s.items()}  # vf2 update
        train_scalars |= {f"times/vfm_{k}": v for k, v in vf_meta_times_s.items()}  # vf-meta update

        return train_scalars

    # ====== Iteration ============================================================

    def _set_iteration(self, it: int):
        pass  # Subclasses may implement this, e.g. for controlling exploration parameters

    def _train_iteration(self, it: int):

        # prep
        self._set_iteration(it)  # informs the algorithm of the current iteration, e.g. for epsilon decay

        # COLLECT ROLLOUTS
        self._buffer.reset()
        with TimedBlock() as rl_rollout_timer:
            rollout_ep_summaries = self._algo.run_environment(env_mode="training",
                                                              scenarios=self._train_scenario_generator.generate(),
                                                              actor_loc=self.train_actor_loc,
                                                              actor_input=self.train_actor_input,
                                                              reset_func=self._reset_env,
                                                              get_action_func=self.get_rollout_action,
                                                              post_step_func=self._post_rollout_step,
                                                              )
        # TRAINING STEP
        self._set_train(True)
        with TimedBlock() as rl_train_timer:
            train_step_summary = self._train_step()
        self._set_train(False)

        # finalize
        iteration_log = {
            "rollout": rollout_ep_summaries,
            "training_step": train_step_summary,
            "rollout_time_s": round(rl_rollout_timer.elapsed_s, 3),
            "training_time_s": round(rl_train_timer.elapsed_s, 3),
        }
        return iteration_log
