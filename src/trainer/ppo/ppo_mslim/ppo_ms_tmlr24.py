from torch.distributions import LogNormal, Normal

import torch
from ._ppo_ms import PPOMSlimTrainer
from src.algorithm.trainable.mslim.mslim import MSlimAlgorithm, TMLR24MSlimAlgorithm, NewMSlimAlgorithm
from src.utils.observations import Observation, ObservationBatch, ObservationGraph
from src.utils.types import ActionRaw, ActionSedi, LogprobTensor, EntropyTensor


TMLR24_EXPLORATION_COEFF = -4.0  # Value taken from TMLR'24 paper


class PPOMSlimTMLR24Trainer(PPOMSlimTrainer):

    def _sample_link_weights(self, actor_input: ObservationGraph) -> (ActionRaw, LogprobTensor):
        """
        Obtain link weights by sampling from a gaussian distribution whose mean is obtained by the actor module
        """
        # obtain mu from actor; use old sigma from TMLR'24 paper
        sigma = torch.exp(torch.tensor(TMLR24_EXPLORATION_COEFF, device=self._algo.actor.device))
        if isinstance(self._algo, TMLR24MSlimAlgorithm):
            mu = self._algo.actor(actor_input).squeeze(-1)
        elif isinstance(self._algo, NewMSlimAlgorithm):
            actor_output = self._algo.actor(actor_input)
            mu, _ = self._algo._split_actor_output(actor_output)
        else:
            raise ValueError("unknown algorithm type")

        if self._algo.pred_in_flat_space:
            weights_dist = Normal(mu, sigma.expand_as(mu))
            link_weights_ = weights_dist.sample()
            logprob = weights_dist.log_prob(link_weights_)
            link_weights = self._algo.softplus(link_weights_)
        else:
            weights_dist = LogNormal(mu, sigma.expand_as(mu))
            link_weights = weights_dist.rsample()
            logprob = weights_dist.log_prob(link_weights)

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
        # obtain mu from actor; use old sigma from TMLR'24 paper
        sigma = torch.exp(torch.tensor(TMLR24_EXPLORATION_COEFF, device=self._algo.actor.device))
        if isinstance(self._algo, TMLR24MSlimAlgorithm):
            mu = self._algo.actor(actor_input).squeeze(-1)
        elif isinstance(self._algo, NewMSlimAlgorithm):
            actor_output = self._algo.actor(actor_input)
            mu, _ = self._algo._split_actor_output(actor_output)
        else:
            raise ValueError("unknown algorithm type")

        if self._algo.pred_in_flat_space:
            weight_dist = Normal(mu, sigma.expand_as(mu))
            logprob = weight_dist.log_prob(self._algo.inv_map_link_weights(rollout_action_raw))
        else:
            weight_dist = LogNormal(mu, sigma.expand_as(mu))
            logprob = weight_dist.log_prob(rollout_action_raw)

        entropy = weight_dist.entropy()
        target_entropy = None  # Not used for this design

        return logprob, entropy, target_entropy
