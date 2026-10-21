import torch
from torch import Tensor

from src.utils.observations import ObservationGraph, Observation, ObservationBatch
from src.utils.types import ActionRaw, ActionSedi, LogprobTensor
from .._ppo import PPOTrainer
from src.algorithm.trainable.mslim import MSlimAlgorithm


class PPOMSlimTrainer(PPOTrainer):

    def _get_value_func_feat_counts(self, feat_counts: dict):
        # self._algo is MSlimAlgorithm
        assert isinstance(self._algo, MSlimAlgorithm)
        if self._algo.link_weights_as_input:
            feat_counts["edge"] -= 1  # last link weights will be removed from the value input
        return feat_counts

    def _get_value_view(self, obs: Observation) -> Observation:
        assert isinstance(self._algo, MSlimAlgorithm)
        return obs.get_value_input(value_loc=self.train_value_loc,
                                   value_input=self.train_value_input,
                                   link_weights_as_input=self._algo.link_weights_as_input)

    def _sample_link_weights(self, actor_input: ObservationGraph) -> (ActionRaw, LogprobTensor):
        """
        Samples the link weights (and connected logprobs from a LogNormal distribution with mu and sigma from policy
        """
        raise NotImplementedError

    def _sample_action(self, obs_graph: ObservationGraph) -> (ActionRaw, ActionSedi, LogprobTensor):
        assert isinstance(self._algo, MSlimAlgorithm)
        actor_input = obs_graph.clone().to(self._algo.actor.device)
        if self._algo.use_line_graph:
            actor_input = actor_input.to_line_graph(self._algo.line_graph_node_to_edge_features)
        link_weights, logprob = self._sample_link_weights(actor_input)
        action_sedi = self._algo._get_action_sedi(link_weights, obs_graph)
        self._algo.set_last_link_weights(obs_graph, link_weights.detach().clone().cpu())
        return link_weights, action_sedi, logprob

    def _get_logprob_mask(self, rollout_obs_batch_view: ObservationBatch):
        # For link-weight policies, it is hard to determine what to include in the logprob mask since the
        # policy outputs link weights, and all of these could in theory be relevant for the resulting
        # local routing even if only parts of that routing will flow into the final global routing action tensor.
        # Therefore, for now, we need to assume that all edges for all observers are valid.

        # E elements for birdseye/central graph for 'birdseye'/'central' actor_input,
        # E elements per local graph for 'local' actor_input
        full_mask = torch.ones_like(rollout_obs_batch_view.observer_ids, dtype=torch.bool)
        return full_mask.repeat_interleave(rollout_obs_batch_view.num_edges_per_graph)

    def _get_logprob_graph_idx(self, rollout_obs_batch_view: ObservationBatch):
        idx_range = torch.arange(rollout_obs_batch_view.num_graphs, device=rollout_obs_batch_view.batch.device)
        return idx_range.repeat_interleave(rollout_obs_batch_view.num_edges_per_graph)

    def _get_entropy_mask(self, rollout_obs_batch_view: ObservationBatch) -> Tensor:
        """
        ~= get_action_raw_mask
        """
        E_per_graph = rollout_obs_batch_view.num_edges_per_graph
        observer_ids_per_edge = rollout_obs_batch_view.observer_ids.repeat_interleave(E_per_graph)
        if self.train_actor_loc == "single":
            # The action stems from a single observation (graph) -> all entries are valid
            return torch.ones_like(observer_ids_per_edge, dtype=torch.bool)
        else:  # self.train_actor_loc == "multi"
            # The action stems from per-agent observation graphs -> we take only those raw values of the action
            # where the agent ID of respective edge source node corresponds to the graph observer
            return observer_ids_per_edge == rollout_obs_batch_view.edge_agent_index_without_offset[0]
