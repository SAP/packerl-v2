import math

import torch
from torch.distributions import LogNormal, Normal

from ._ppo_ms import PPOMSlimTrainer
from src.algorithm.trainable.mslim.mslim import MSlimAlgorithm, NewMSlimAlgorithm, TMLR24MSlimAlgorithm
from src.utils.observations import ObservationGraph, ObservationBatch
from src.utils.types import ActionRaw, ActionSedi, LogprobTensor, EntropyTensor


class PPOMSlimNewTrainer(PPOMSlimTrainer):

    target_entropy_per_edge: float  # Target entropy per edge in the graph, initialized in abstract trainer init

    def _sample_link_weights(self, actor_input: ObservationGraph) -> (ActionRaw, LogprobTensor):
        """
        Samples the link weights (and connected logprobs from a LogNormal distribution with mu and sigma from policy
        """
        assert isinstance(self._algo, MSlimAlgorithm)

        if isinstance(self._algo, TMLR24MSlimAlgorithm):
            mu = self._algo.actor(actor_input).squeeze(-1)
            sigma = torch.ones_like(mu)  # TODO make this learnable ?
        elif isinstance(self._algo, NewMSlimAlgorithm):
            actor_output = self._algo.actor(actor_input)
            mu, sigma = self._algo._split_actor_output(actor_output)
        else:
            raise ValueError("unknown algorithm type")

        if self._algo.pred_in_flat_space:
            weight_dist = Normal(mu, sigma)
            link_weights_ = weight_dist.sample()
            logprob = weight_dist.log_prob(link_weights_)
            link_weights = self._algo.softplus(link_weights_)
        else:
            weight_dist = LogNormal(mu, sigma)
            link_weights = weight_dist.rsample()
            logprob = weight_dist.log_prob(link_weights)

        return link_weights, logprob

    def _evaluate_action(self,
                         rollout_obs_batch_view: ObservationBatch,
                         rollout_action_raw: ActionRaw,
                         _: ActionSedi) -> (LogprobTensor, EntropyTensor, EntropyTensor):
        """
        Evaluates the action with respect to the given observation batch.
        """
        assert isinstance(self._algo, MSlimAlgorithm)
        if self._algo.use_line_graph:
            actor_input = (rollout_obs_batch_view.clone().to(self._algo.actor.device)
                           .to_line_graph(self._algo.line_graph_node_to_edge_features))
        else:
            actor_input = rollout_obs_batch_view.clone().to(self._algo.actor.device)
        if isinstance(self._algo, TMLR24MSlimAlgorithm):
            mu = self._algo.actor(actor_input).squeeze(-1)
            sigma = torch.ones_like(mu)  # TODO make this learnable ?
        elif isinstance(self._algo, NewMSlimAlgorithm):
            actor_output = self._algo.actor(actor_input)
            mu, sigma = self._algo._split_actor_output(actor_output)
        else:
            raise ValueError("unknown algorithm type")

        if self._algo.pred_in_flat_space:
            weight_dist = Normal(mu, sigma)
            entropy = 0.5 + 0.5 * math.log(2.0 * math.pi) + torch.log(sigma)
            logprob = weight_dist.log_prob(self._algo.inv_map_link_weights(rollout_action_raw))
        else:
            weight_dist = LogNormal(mu, sigma)  # Distribution under current policy
            entropy = mu + torch.log(sigma) + 0.5 * (1.0 + math.log(2.0 * math.pi))
            logprob = weight_dist.log_prob(rollout_action_raw)

        target_entropy = torch.ones_like(entropy) * self.target_entropy_per_edge
        return logprob, entropy, target_entropy
